"""واجهة Streamlit عصرية: بث مباشر فيديو (WebRTC) / صور / فيديو + بناء جمل + ترجمة + نطق."""
from __future__ import annotations

import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from src.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_IOU
from src.detector import class_names, load_model, predict
from src.sentence_builder import SentenceBuilder
from src.translator import translate, tts_bytes

# ============================ Page setup =============================
st.set_page_config(
    page_title="مترجم لغة الإشارة العربية",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================ Theme / CSS ============================
def inject_css(dark: bool) -> None:
    if dark:
        bg, surface, text, muted, border, accent = (
            "#0b1020", "#131a2e", "#f5f7fb", "#9aa4bf", "#1f2942", "#7c5cff"
        )
    else:
        bg, surface, text, muted, border, accent = (
            "#f6f7fb", "#ffffff", "#0f172a", "#4b5568", "#e5e7eb", "#6d4bff"
        )

    st.markdown(
        f"""
        <style>
        html, body, [class*="stApp"], .stApp [data-testid="stAppViewContainer"], header[data-testid="stHeader"] {{
            background: {bg} !important;
        }}
        
        section[data-testid="stSidebar"] {{
            background: {surface} !important;
            border-left: 1px solid {border} !important;
        }}

        p, h1, h2, h3, h4, h5, h6, span, label, li {{
            color: {text} !important;
        }}

        .hero p, .hero h1, .hero span {{
            color: white !important;
        }}

        div[data-baseweb="select"] > div {{
            background-color: {surface} !important;
            border-color: {border} !important;
        }}
        div[data-baseweb="popover"] > div {{
            background-color: {surface} !important;
        }}

        .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1300px; }}
        .hero {{
            background: linear-gradient(135deg, {accent} 0%, #22d3ee 100%);
            border-radius: 22px; padding: 22px 26px; color: white;
            box-shadow: 0 10px 40px -12px {accent}66;
            display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;
        }}
        .hero h1 {{ margin:0; font-size: 1.6rem; font-weight: 800; }}
        .hero p {{ margin: 4px 0 0; opacity: .9; font-size: .95rem; }}
        .badge {{
            background: rgba(255,255,255,.18); padding: 6px 12px;
            border-radius: 999px; font-size:.8rem; backdrop-filter: blur(6px);
        }}
        .sentence-box {{
            background: {surface} !important; border: 2px dashed {accent}55 !important;
            border-radius: 18px; padding: 24px; min-height: 90px;
            font-size: 2rem; font-weight: 700; 
            direction: rtl; text-align: right; line-height: 1.6;
            word-wrap: break-word;
        }}
        .translation-box {{
            background: linear-gradient(135deg,{accent}15,transparent);
            border-radius: 14px; padding: 16px; 
            font-size: 1.15rem; direction: ltr; text-align: left;
            border: 1px solid {border} !important;
        }}
        .chip {{
            display:inline-block; padding: 6px 12px; margin: 3px;
            background:{accent}22; border-radius: 999px;
            font-size: .85rem; border:1px solid {accent}44;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            background: {surface} !important; border-radius: 12px; padding: 10px 18px;
            border:1px solid {border} !important;
        }}
        .stTabs [data-baseweb="tab"] p {{
            color: {text} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent} !important; border-color: {accent} !important;
        }}
        .stTabs [aria-selected="true"] p {{
            color: white !important;
        }}

        .stButton > button {{
            border-radius: 12px; border: 1px solid {border} !important; 
            background: {surface} !important; 
            transition: all .15s ease;
        }}
        .stButton > button p {{
            color: {text} !important;
        }}
        .stButton > button:hover {{
            border-color: {accent} !important; transform: translateY(-1px);
        }}
        .stButton > button:hover p {{
            color: {accent} !important;
        }}

        div[data-testid="stFileUploader"] section {{
            background: {surface} !important; border: 2px dashed {border} !important; border-radius: 14px;
        }}
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )


# ============================ State ==================================
if "dark" not in st.session_state:
    st.session_state.dark = True
if "builder" not in st.session_state:
    st.session_state.builder = SentenceBuilder()

builder: SentenceBuilder = st.session_state.builder


# ============================ WebRTC Video Processor =================
class SignLanguageProcessor(VideoProcessorBase):
    """معالج الفريمات المباشر للبث الحي."""

    def __init__(self) -> None:
        self.conf = DEFAULT_CONF
        self.iou = DEFAULT_IOU
        self.imgsz = DEFAULT_IMGSZ
        self.builder: SentenceBuilder | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # التنبؤ بالفريم الحي عبر YOLO
        results = predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        res = results[0]
        annotated = res.plot()

        if len(res.boxes) and self.builder is not None:
            i = int(np.argmax(res.boxes.conf.cpu().numpy()))
            label = res.names[int(res.boxes.cls[i])]
            cf = float(res.boxes.conf[i])
            self.builder.observe(label, cf)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ============================ Sidebar ================================
with st.sidebar:
    st.markdown("### ⚙️ الإعدادات")
    st.session_state.dark = st.toggle("🌙 الوضع الليلي", value=st.session_state.dark)

    st.markdown("#### دقّة الكشف")
    conf = st.slider("Confidence", 0.05, 0.95, DEFAULT_CONF, 0.05)
    iou = st.slider("IoU", 0.1, 0.95, DEFAULT_IOU, 0.05)
    imgsz = st.select_slider("Image size", [320, 480, 640, 800, 960], DEFAULT_IMGSZ)

    st.markdown("#### بناء الجمل")
    builder.stability_frames = st.slider(
        "ثبات الحرف (frames)", 2, 20, builder.stability_frames
    )
    builder.min_confidence = st.slider(
        "أقل ثقة للقبول", 0.1, 0.95, builder.min_confidence, 0.05
    )

    st.divider()
    target_lang = st.selectbox(
        "🌍 لغة الترجمة",
        options=["en", "fr", "es", "de", "tr", "it", "ru", "zh-CN"],
        format_func=lambda x: {
            "en": "English", "fr": "Français", "es": "Español", "de": "Deutsch",
            "tr": "Türkçe", "it": "Italiano", "ru": "Русский", "zh-CN": "中文",
        }[x],
    )

    st.divider()
    try:
        load_model()
        st.success(f"✅ الموديل جاهز — {len(class_names())} فئة")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

inject_css(st.session_state.dark)


# ============================ Hero ===================================
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>🤟 مترجم لغة الإشارة العربية</h1>
        <p>YOLO · بث مباشر للبث الحي · بناء تلقائي · نطق وترجمة</p>
      </div>
      <div>
        <span class="badge">🎥 Live Stream WebRTC</span>
        <span class="badge">🧠 {len(class_names())} فئة</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ============================ Sentence panel =========================
def sentence_panel(key_prefix: str) -> None:
    st.markdown("#### ✍️ الجملة المُكوَّنة")
    
    placeholder = '<span style="opacity:.4">ابدأ بالإشارة أمام الكاميرا…</span>'
    display_text = builder.text or placeholder
    
    st.markdown(
        f"<div class='sentence-box'>{display_text}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("␣ مسافة", use_container_width=True, key=f"{key_prefix}_space"):
        builder.add_space(); st.rerun()
    if c2.button("⌫ حذف", use_container_width=True, key=f"{key_prefix}_del"):
        builder.backspace(); st.rerun()
    if c3.button("🗑️ مسح الكل", use_container_width=True, key=f"{key_prefix}_clear"):
        builder.clear(); st.rerun()
    if c4.button("🔊 نطق عربي", use_container_width=True, key=f"{key_prefix}_audio") and builder.text.strip():
        try:
            st.audio(tts_bytes(builder.text, "ar"), format="audio/mp3")
        except Exception as e:
            st.error(f"خطأ في النطق: {e}")
    if c5.button("🌐 ترجم", use_container_width=True, key=f"{key_prefix}_trans") and builder.text.strip():
        st.session_state._translation = translate(builder.text, target=target_lang, source="ar")

    if builder.words:
        st.markdown("**الكلمات:** " + "".join(f"<span class='chip'>{w}</span>" for w in builder.words),
                    unsafe_allow_html=True)

    if st.session_state.get("_translation"):
        st.markdown(
            f"<div class='translation-box'>🌍 {st.session_state._translation}</div>",
            unsafe_allow_html=True,
        )
        try:
            st.audio(tts_bytes(st.session_state._translation, target_lang), format="audio/mp3")
        except Exception:
            pass


# ============================ Helpers ================================
def _top_detection(result) -> tuple[str | None, float]:
    if not len(result.boxes):
        return None, 0.0
    i = int(np.argmax(result.boxes.conf.cpu().numpy()))
    return result.names[int(result.boxes.cls[i])], float(result.boxes.conf[i])


# ============================ Tabs ===================================
tab_cam, tab_img, tab_vid, tab_about = st.tabs(
    ["🎥 بث مباشر فيديو", "🖼️ صورة", "🎬 فيديو", "ℹ️ عن المشروع"]
)

# ---------------- Live Streaming (WebRTC) ----------------
with tab_cam:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("#### 🎥 البث المباشر للكاميرا")
        st.caption("اضغط START لبدء البث المباشر بالكاميرا واكتشاف الإشارات فوريًا.")

        ctx = webrtc_streamer(
            key="arsl-live-stream",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration(
                {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:global.stun.twilio.com:3478"]},
                    ]
                }
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.conf = conf
            ctx.video_processor.iou = iou
            ctx.video_processor.imgsz = imgsz
            ctx.video_processor.builder = builder

    with col2:
        sentence_panel("cam")

# ---------------- Image ----------------
with tab_img:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        up = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if up:
            img = np.array(Image.open(up).convert("RGB"))
            results = predict(img[:, :, ::-1], conf=conf, iou=iou, imgsz=imgsz)
            res = results[0]
            annotated = res.plot()[:, :, ::-1]
            st.image(annotated, caption="النتيجة", use_column_width=True)

            label, cf = _top_detection(res)
            if label:
                builder.observe(label, cf)
                st.success(f"✅ تم الاكتشاف: **{label}** ({cf:.2%})")
            else:
                st.warning("لم يتم اكتشاف أي إشارة.")
    with col2:
        sentence_panel("img")

# ---------------- Video ----------------
with tab_vid:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        up = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])
        if up:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tf.write(up.read()); tf.close()
            frame_slot = st.empty()
            for result in predict(tf.name, conf=conf, iou=iou, imgsz=imgsz, stream=True):
                frame_slot.image(result.plot()[:, :, ::-1], channels="RGB")
                label, cf = _top_detection(result)
                if label:
                    builder.observe(label, cf)
            st.success("✅ تم الانتهاء من معالجة الفيديو")
    with col2:
        sentence_panel("vid")

# ---------------- About ----------------
with tab_about:
    st.markdown("""
### 🤟 مترجم لغة الإشارة العربية (بث مباشر)
تطبيق ذكي لاكتشاف حروف لغة الإشارة العربية عبر البث المباشر للفيديو، وبناء الكلمات والجمل مع النطق والترجمة.
""")
