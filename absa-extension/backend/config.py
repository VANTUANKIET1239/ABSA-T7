from dataclasses import dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASPECTS = (
    ("CleansingEffectiveness", "Hiệu quả làm sạch"),
    ("HydrationSkinComfort", "Cấp ẩm và độ dễ chịu"),
    ("SkinCompatibilitySafety", "Mức độ an toàn"),
    ("TextureFoaming", "Kết cấu và tạo bọt"),
    ("Fragrance", "Mùi hương"), ("IngredientsFormulation", "Thành phần"),
    ("Packaging", "Bao bì"), ("PriceValue", "Giá cả"),
    ("ShopService", "Dịch vụ cửa hàng"), ("Delivery", "Giao hàng"),
    ("General", "Đánh giá tổng quan"),
)
POLARITIES = ("Positive", "Negative", "Neutral", "None")


@dataclass(frozen=True)
class Settings:
    checkpoint: Path = ROOT / "train-models/Models/Cleanser/best_model_birnn_cleanser_focal_loss_VI.pt"
    model_name: str = "vinai/phobert-base"
    research_root: Path = ROOT
    vncorenlp_dir: Path = ROOT / "vncorenlp_model"
    java: str = "java"
    device: str = "auto"
    batch_size: int = 16
    max_reviews: int = 100
    max_review_chars: int = 4000
    max_total_chars: int = 150000
    max_length: int = 128
    torch_threads: int = 4
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        defaults = cls()
        research_root = Path(os.environ.get("ABSA_RESEARCH_ROOT", str(ROOT))).expanduser()
        values = dict(checkpoint=Path(os.environ.get("ABSA_CHECKPOINT", str(research_root /
                      "train-models/Models/Cleanser/best_model_birnn_cleanser_focal_loss_VI.pt"))).expanduser(),
                      model_name=os.environ.get("ABSA_MODEL_NAME", defaults.model_name),
                      research_root=research_root,
                      vncorenlp_dir=Path(os.environ.get("VNCORENLP_DIR", str(research_root / "vncorenlp_model"))).expanduser(),
                      java=os.environ.get("JAVA_EXECUTABLE", "java"), device=os.environ.get("ABSA_DEVICE", "auto"))
        values["allowed_hosts"] = tuple(host.strip() for host in os.environ.get("ABSA_ALLOWED_HOSTS", ",".join(defaults.allowed_hosts)).split(",") if host.strip())
        secret_file = os.environ.get("ABSA_API_KEY_FILE")
        values["api_key"] = Path(secret_file).read_text(encoding="utf-8").strip() if secret_file else os.environ.get("ABSA_API_KEY", "").strip()
        if secret_file and not values["api_key"]:
            raise ValueError("API key file is empty.")
        for field, variable in (("batch_size", "ABSA_BATCH_SIZE"), ("max_reviews", "ABSA_MAX_REVIEWS"),
                                ("max_review_chars", "ABSA_MAX_REVIEW_CHARS"),
                                ("max_total_chars", "ABSA_MAX_TOTAL_CHARS"), ("torch_threads", "ABSA_TORCH_THREADS")):
            value = int(os.environ.get(variable, getattr(defaults, field)))
            if value < 1:
                raise ValueError(f"{variable} phải lớn hơn 0.")
            values[field] = value
        return cls(**values)
