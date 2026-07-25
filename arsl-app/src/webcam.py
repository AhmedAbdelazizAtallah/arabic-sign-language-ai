"""تشغيل real-time detection من الكاميرا مع FPS overlay."""
from __future__ import annotations

import time

import cv2

from .detector import class_names, load_model, predict
from .config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU


def run_webcam(
    camera_index: int = 0,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    mirror: bool = True,
) -> None:
    load_model()  # warmup قبل ما نفتح الكاميرا
    print(f"[info] Classes: {class_names()}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"مش قادر أفتح الكاميرا رقم {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev = time.time()
    fps = 0.0
    win = "Arabic Sign Language - YOLO (Q to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            results = predict(frame, conf=conf, iou=iou, imgsz=imgsz)
            annotated = results[0].plot()

            now = time.time()
            dt = now - prev
            prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(win, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()