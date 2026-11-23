"""Exact research BiRNN head and label mapping; full checkpoint is loaded strictly."""
import hashlib
import threading
try:
    from .config import ASPECTS, POLARITIES
    from .preprocessing import ResearchPreprocessor
    from .schemas import AspectPrediction, ReviewResult
except ImportError:
    from config import ASPECTS, POLARITIES
    from preprocessing import ResearchPreprocessor
    from schemas import AspectPrediction, ReviewResult


class ModelUnavailable(RuntimeError):
    pass


def build_research_model(torch, config):
    from transformers import AutoModel
    nn = torch.nn

    class PhoBERT_BiRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert = AutoModel.from_config(config)
            self.birnn = nn.RNN(input_size=768, hidden_size=256, batch_first=True,
                                bidirectional=True, nonlinearity="tanh")
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(512, 11 * 4)

        def forward(self, input_ids, attention_mask):
            sequence = self.phobert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            recurrent, _ = self.birnn(sequence)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(recurrent * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            return self.fc(self.dropout(pooled)).view(-1, 11, 4)

    return PhoBERT_BiRNN()


def decode_probabilities(probabilities):
    """Indices 0/1/2/3 = Positive/Negative/Neutral/None; None never becomes neutral."""
    predictions = []
    if len(probabilities) != 11 or any(len(row) != 4 for row in probabilities):
        raise ValueError("Expected probabilities shaped [11, 4].")
    for index, (values, (aspect, name)) in enumerate(zip(probabilities, ASPECTS), 1):
        label = max(range(4), key=lambda i: values[i])
        if label != 3:
            predictions.append(AspectPrediction(code=f"A{index}", aspect=aspect, name=name,
                                               sentiment=POLARITIES[label], confidence=float(values[label])))
    return predictions


class ModelService:
    def __init__(self, settings):
        self.settings = settings
        self.loaded = False
        self.device = None
        self.version = ""
        self.lock = threading.Lock()
        self.preprocessor = ResearchPreprocessor(settings)

    def load(self):
        if self.loaded:
            return
        if not self.settings.checkpoint.is_file():
            raise ModelUnavailable("Không tìm thấy checkpoint. Cấu hình ABSA_CHECKPOINT tới BiRNN Focal Loss.")
        import torch
        from transformers import AutoConfig, AutoTokenizer
        self.preprocessor.load()
        torch.set_num_threads(self.settings.torch_threads)
        try:
            config = AutoConfig.from_pretrained(self.settings.model_name, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name, local_files_only=True)
        except (OSError, ValueError) as exc:
            raise ModelUnavailable("Thiếu config/tokenizer PhoBERT trong cache local; kiểm tra ABSA_MODEL_NAME.") from exc
        if config.hidden_size != 768:
            raise ModelUnavailable("Config không khớp PhoBERT hidden_size=768.")
        device = self.settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelUnavailable("Đã yêu cầu CUDA nhưng PyTorch chưa nhận GPU.")
        self.device = str(torch.device(device))
        model = build_research_model(torch, config)
        try:
            weights = torch.load(self.settings.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ModelUnavailable("Checkpoint không khớp kiến trúc PhoBERT–BiRNN 11 × 4. Không tự chuyển mô hình.") from exc
        del weights
        self.model = model.to(self.device).eval()
        self.torch = torch
        with self.settings.checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        preprocessing = self.settings.research_root / "reprocessing/preprocess_cleanser_reviews.py"
        prep_hash = hashlib.sha256(preprocessing.read_bytes()).hexdigest()[:12]
        self.version = f"{self.settings.checkpoint.stem}:{checkpoint_hash[:12]}:len128:prep{prep_hash}"
        self.loaded = True

    def analyze(self, texts):
        if not self.loaded:
            raise ModelUnavailable("Mô hình chưa sẵn sàng. Kiểm tra /health.")
        with self.lock:
            cleaned = self.preprocessor.process(texts)
            results = [ReviewResult(review_id=i, text=text, aspects=[], status="analyzed" if cleaned[i] else "empty")
                       for i, text in enumerate(texts)]
            positions = [i for i, text in enumerate(cleaned) if text]
            with self.torch.inference_mode():
                for start in range(0, len(positions), self.settings.batch_size):
                    batch = positions[start:start + self.settings.batch_size]
                    encoded = self.tokenizer([cleaned[i] for i in batch], max_length=128, padding="max_length",
                                             truncation=True, return_tensors="pt")
                    logits = self.model(input_ids=encoded["input_ids"].to(self.device),
                                        attention_mask=encoded["attention_mask"].to(self.device))
                    if tuple(logits.shape) != (len(batch), 11, 4):
                        raise ModelUnavailable("Đầu ra model không đúng kích thước batch × 11 × 4.")
                    probabilities = logits.softmax(dim=-1).cpu().tolist()
                    for position, values in zip(batch, probabilities):
                        results[position].aspects = decode_probabilities(values)
            return results
