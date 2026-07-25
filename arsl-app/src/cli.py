"""واجهة سطر أوامر."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU, OUTPUT_DIR
from .detector import predict
from .webcam import run_webcam


def _cmd_image(args: argparse.Namespace) -> None:
    results = predict(args.input, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    out_path = OUTPUT_DIR / f"pred_{Path(args.input).name}"
    cv2.imwrite(str(out_path), results[0].plot())
    print(f"[ok] Saved: {out_path}")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  - {results[0].names[cls_id]}  ({conf:.2f})")


def _cmd_video(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"مش قادر أفتح الفيديو: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = OUTPUT_DIR / f"pred_{Path(args.input).stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    try:
        for result in predict(
            args.input, conf=args.conf, iou=args.iou, imgsz=args.imgsz, stream=True
        ):
            writer.write(result.plot())
    finally:
        writer.release()
        cap.release()
    print(f"[ok] Saved: {out_path}")


def _cmd_webcam(args: argparse.Namespace) -> None:
    run_webcam(
        camera_index=args.camera,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        mirror=not args.no_mirror,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arsl", description="Arabic Sign Language Detector (YOLOv8)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--conf", type=float, default=DEFAULT_CONF)
    common.add_argument("--iou", type=float, default=DEFAULT_IOU)
    common.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)

    p_img = sub.add_parser("image", parents=[common], help="تنبؤ على صورة واحدة")
    p_img.add_argument("input", type=str)
    p_img.set_defaults(func=_cmd_image)

    p_vid = sub.add_parser("video", parents=[common], help="تنبؤ على ملف فيديو")
    p_vid.add_argument("input", type=str)
    p_vid.set_defaults(func=_cmd_video)

    p_cam = sub.add_parser("webcam", parents=[common], help="تشغيل الكاميرا")
    p_cam.add_argument("--camera", type=int, default=0)
    p_cam.add_argument("--no-mirror", action="store_true")
    p_cam.set_defaults(func=_cmd_webcam)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()