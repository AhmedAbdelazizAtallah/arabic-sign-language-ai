"""واجهة Streamlit اختيارية: رفع صور/فيديو + كاميرا داخل المتصفح."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU
from src.detector import class_names, load_model, predict

st.set_page_config(
    page_title="Arabic Sign Language Detector",
    page_icon="🤟",
    layout="wide",
)

st.title("🤟 مترجم لغة الإشارة العربية")
st.caption("YOLOv8 detection — ارفع صورة/فيديو أو استخدم الكاميرا")

with st.sidebar:
    st.header("الإعدادات")
    conf = st.slider("Confidence", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou = st.slider("IoU", 0.1, 0.95, DEFAULT_IOU, 0.05)
    imgsz = st.select_slider("Image size", [320, 480, 640, 800, 960], DEFAULT_IMGSZ)
    st.divider()
    try:
        load_model()
        st.success(f"الموديل جاهز — {len(class_names())} فئة")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ صورة", "🎬 فيديو", "📷 كاميرا"])

with tab_img:
    up = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up:
        img = np.array(Image.open(up).convert("RGB"))
        results = predict(img[:, :, ::-1], conf=conf, iou=iou, imgsz=imgsz)
        annotated = results[0].plot()[:, :, ::-1]
        st.image(annotated, caption="النتيجة", use_column_width=True)
        if len(results[0].boxes):
            st.subheader("العلامات المكتشفة")
            for b in results[0].boxes:
                st.write(
                    f"- **{results[0].names[int(b.cls[0])]}** — {float(b.conf[0]):.2f}"
                )
        else:
            st.info("لم يتم اكتشاف أي إشارة.")

with tab_vid:
    up = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])
    if up:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
        tf.write(up.read())
        tf.close()
        cap = cv2.VideoCapture(tf.name)
        frame_slot = st.empty()
        stop = st.button("إيقاف")
        for result in predict(tf.name, conf=conf, iou=iou, imgsz=imgsz, stream=True):
            if stop:
                break
            frame_slot.image(result.plot()[:, :, ::-1], channels="RGB")
        cap.release()

with tab_cam:
    st.info("للأداء الحقيقي real-time شغّل: `python -m src.cli webcam`")
    snap = st.camera_input("خد صورة من الكاميرا")
    if snap:
        img = np.array(Image.open(snap).convert("RGB"))
        results = predict(img[:, :, ::-1], conf=conf, iou=iou, imgsz=imgsz)
        st.image(results[0].plot()[:, :, ::-1], channels="RGB")
        for b in results[0].boxes:
            st.write(
                f"- **{results[0].names[int(b.cls[0])]}** — {float(b.conf[0]):.2f}"
            )