"""Reuse the research normalization; one local Java process segments each request."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile


class PreprocessingError(RuntimeError):
    pass


def parse_segmented_output(text, expected):
    groups, tokens = [], []
    for line in text.splitlines() + [""]:
        if not line.strip():
            if tokens:
                groups.append(re.sub(r"\s*_\s*", "_", " ".join(tokens)))
                tokens = []
            continue
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0].isdigit():
            tokens.append(fields[1])
    if len(groups) != expected:
        raise PreprocessingError("Không khớp số review sau phân đoạn; không gán nhãn sang review khác.")
    return groups


class ResearchPreprocessor:
    def __init__(self, settings):
        self.settings = settings
        self.module = None

    def load(self):
        path = self.settings.research_root / "reprocessing/preprocess_cleanser_reviews.py"
        if not path.is_file():
            raise PreprocessingError("Thiếu source tiền xử lý nghiên cứu. Kiểm tra ABSA_RESEARCH_ROOT.")
        resources = ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr")
        if any(not (self.settings.vncorenlp_dir / name).is_file() for name in resources):
            raise PreprocessingError("Thiếu tài nguyên VnCoreNLP. Kiểm tra VNCORENLP_DIR.")
        try:
            subprocess.run([self.settings.java, "-version"], capture_output=True, check=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreprocessingError("Không chạy được Java. Kiểm tra JAVA_EXECUTABLE hoặc PATH.") from exc
        # Load the actual research file, without executing its guarded dataset CLI.
        parent = str(path.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec = importlib.util.spec_from_file_location("_absa_research_preprocessing", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def process(self, texts):
        if self.module is None:
            raise PreprocessingError("Tiền xử lý chưa sẵn sàng.")
        normalized = [self.module.apply_phrase_underscore(self.module.normalize_comment(text)) for text in texts]
        positions = [i for i, text in enumerate(normalized) if text.strip()]
        output = [""] * len(texts)
        if not positions:
            return output
        with tempfile.TemporaryDirectory(prefix="absa-extension-") as temporary:
            source, target = Path(temporary) / "input.txt", Path(temporary) / "output.txt"
            source.write_text("\n".join(normalized[i] for i in positions) + "\n", encoding="utf-8")
            try:
                result = subprocess.run([
                    self.settings.java, "-Xmx512m", "-Dfile.encoding=UTF-8", "-jar",
                    str(self.settings.vncorenlp_dir / "VnCoreNLP-1.2.jar"),
                    "-fin", str(source), "-fout", str(target), "-annotators", "wseg"],
                    cwd=self.settings.vncorenlp_dir, capture_output=True, timeout=90,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise PreprocessingError("Java/VnCoreNLP không hoàn thành phân đoạn.") from exc
            if result.returncode or not target.exists():
                raise PreprocessingError("VnCoreNLP không trả kết quả hợp lệ.")
            groups = parse_segmented_output(target.read_text(encoding="utf-8-sig"), len(positions))
            for position, segmented in zip(positions, groups):
                output[position] = self.module.remove_punctuation(segmented)
        return output
