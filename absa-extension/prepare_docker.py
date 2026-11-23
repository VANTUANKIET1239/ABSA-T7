"""Export only offline tokenizer assets and create a private API key once."""
import os
from pathlib import Path
import secrets

APP = Path(__file__).resolve().parent
ROOT = APP.parent


def main():
    required = [ROOT / "train-models/Models/Cleanser/best_model_birnn_cleanser_focal_loss_VI.pt",
                ROOT / "vncorenlp_model/VnCoreNLP-1.2.jar",
                ROOT / "vncorenlp_model/models/wordsegmenter/vi-vocab",
                ROOT / "vncorenlp_model/models/wordsegmenter/wordsegmenter.rdr"]
    if any(not path.is_file() for path in required):
        raise RuntimeError("Missing checkpoint or VnCoreNLP resources.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoConfig, AutoTokenizer
    name = os.environ.get("ABSA_MODEL_NAME", "vinai/phobert-base")
    config = AutoConfig.from_pretrained(name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    if config.hidden_size != 768:
        raise RuntimeError("Expected PhoBERT hidden_size=768.")
    assets = APP / ".docker/assets/phobert"
    assets.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(assets); tokenizer.save_pretrained(assets)
    AutoTokenizer.from_pretrained(assets, local_files_only=True)
    key_file = APP / ".docker/api_key.txt"
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(32), encoding="utf-8")
    if not key_file.read_text(encoding="utf-8").strip():
        raise RuntimeError("API key file is empty.")
    print("Ready: offline tokenizer, read-only model resources and private API key (not printed).")


if __name__ == "__main__":
    main()
