"""إعدادات المشروع المشتركة."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best.pt"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# إعدادات الاستدلال الافتراضية
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640

# استخدم GPU لو متاح، غير كده CPU
def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"