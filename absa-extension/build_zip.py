"""Package source only; omit checkpoints, environments, private config and test caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
SKIP = {".git", ".cache", ".venv", ".docker", "__pycache__", "node_modules", "dist"}

if __name__ == "__main__":
    output = ROOT / "dist/shopee-review-analyzer.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for source in sorted(ROOT.rglob("*")):
            relative = source.relative_to(ROOT)
            if not source.is_file() or SKIP.intersection(relative.parts) or source.name in {".env", "api_config.local.json"}:
                continue
            if source.suffix in {".pt", ".pth", ".safetensors", ".pyc", ".zip"}:
                continue
            archive.write(source, Path(ROOT.name) / relative)
    print(output)
