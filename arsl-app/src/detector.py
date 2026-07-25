"""طبقة تجريد فوق موديل YOLO لتحميله مرة واحدة والتنبؤ بسرعة."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from ultralytics import YOLO

from .config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU, MODEL_PATH, get_device


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> YOLO:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"لم يتم العثور على الموديل في: {path}\n"
            "ضع ملف best.pt داخل arsl-app/models/"
        )
    model = YOLO(str(path))
    # تسخين الموديل (warmup) عشان أول تنبؤ يبقى سريع
    dummy = np.zeros((DEFAULT_IMGSZ, DEFAULT_IMGSZ, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=DEFAULT_IMGSZ, device=get_device(), verbose=False)
    return model


def predict(
    source,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    stream: bool = False,
) -> Iterable:
    """تنبؤ عام. source ممكن يكون path صورة/فيديو أو ndarray أو رقم كاميرا."""
    model = load_model()
    return model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=get_device(),
        stream=stream,
        verbose=False,
    )


def class_names() -> dict[int, str]:
    return load_model().names