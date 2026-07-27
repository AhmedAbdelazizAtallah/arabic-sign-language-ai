"""واجهة Streamlit عصرية: كاميرا مباشر (WebRTC) / صور / فيديو + بناء جمل + ترجمة + نطق."""
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
        /* إجبار الخلفية الأساسية والشريط العلوي */
        html, body, [class*="stApp"], .stApp [data-testid="stAppViewContainer"], header[data-testid="stHeader"] {{
            background: {bg} !important;
        }}
        
        /* إجبار خلفية الشريط الجانبي (Sidebar) */
        section[data-testid="stSidebar"] {{
            background: {surface} !important;
            border-left: 1px solid {border} !important;
        }}

        /* إجبار لون النص على كل العناصر لتخطي ألوان Streamlit الافتراضية */
        p, h1, h2, h3, h4, h5, h6, span, label, li {{
            color: {text} !important;
        }}

        /* استثناء منطقة الـ Hero للحفاظ على نصوصها بيضاء دائماً */
        .hero p, .hero h1, .hero span {{
            color: white !important;
        }}

        /* تحسين شكل القائمة المنسدلة للغات (Selectbox) */
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
        .card {{
            background: {surface} !important; border: 1px solid {border} !important;
            border-radius: 18px; padding: 18px; margin-bottom: 14px;
            box-shadow: 0 4px 20px -12px rgba(0,0,0,.15);
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
        
        /* تعديل التبويبات (Tabs) */
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

        /* تعديل الأزرار (Buttons) */
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
        .stAlert {{ border-radius: 12px; }}
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
    """معالج الفريمات للبث المباشر عبر WebRTC."""

    def __init__(self) -> None:
        self.conf = DEFAULT_CONF
        self.iou = DEFAULT_IOU
        self.imgsz = DEFAULT_IMGSZ
        self.builder: SentenceBuilder | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # التنبؤ بواسطة YOLO
        results = predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        res = results[0]
        annotated = res.plot()

        # بناء الجملة تلقائياً إذا تم اكتشاف إشارة
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
    builder.cooldown_frames = st.slider(
        "وقت الانتظار بعد الحرف", 0, 20, builder.cooldown_frames
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
        <p>YOLOv26 · بناء تلقائي للكلمات والجمل · نطق وترجمة فورية</p>
      </div>
      <div>
        <span class="badge">⚡ Real-time WebRTC</span>
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
    st.markdown(
        f"<div class='sentence-box'>{builder.text or '<span style=\"opacity:.4\">ابدأ بالإشارة…</span>'}</div>",
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
        except Exception as e:  # noqa: BLE001
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

    if builder.text.strip():
        st.download_button(
            "⬇️ تحميل الجملة (.txt)",
            data=builder.text.encode("utf-8"),
            file_name="sentence.txt",
            mime="text/plain",
            key=f"{key_prefix}_download"
        )


# ============================ Helpers ================================
def _top_detection(result) -> tuple[str | None, float]:
    if not len(result.boxes):
        return None, 0.0
    i = int(np.argmax(result.boxes.conf.cpu().numpy()))
    return result.names[int(result.boxes.cls[i])], float(result.boxes.conf[i])


# ============================ Tabs ===================================
tab_cam, tab_img, tab_vid, tab_about = st.tabs(
    ["📷 كاميرا مباشر", "🖼️ صورة", "🎬 فيديو", "ℹ️ عن المشروع"]
)

# ---------------- Live camera (streamlit-webrtc) ----------------
with tab_cam:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("#### 🎥 البث المباشر للكاميرا")
        st.caption("اضغط START للسماح بالكاميرا والبدء في الكشف المباشر في الوقت الفعلي.")

        ctx = webrtc_streamer(
            key="arsl-webrtc-stream",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # تمرير الإعدادات و dynamic parameters للمعالج في الوقت الفعلي
        if ctx.video_processor:
            ctx.video_processor.conf = conf
            ctx.video_processor.iou = iou
            ctx.video_processor.imgsz = imgsz
            ctx.video_processor.builder = builder

    with col2:
        sentence_panel("cam")

    st.info("💡 يتم بث الفيديو مباشرة عبر WebRTC لمعالجة فائقة السرعة واستجابة فورية.")

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

            if len(res.boxes):
                rows = []
                for b in res.boxes:
                    rows.append({
                        "الحرف": res.names[int(b.cls[0])],
                        "الثقة": f"{float(b.conf[0]):.2%}",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                label, cf = _top_detection(res)
                if label:
                    for _ in range(builder.stability_frames):
                        added = builder.observe(label, cf)
                    if added:
                        st.success(f"✅ أضيف تلقائياً: **{added}** ({cf:.2f})")
                    else:
                        st.info(f"🔍 مكتشف: **{label}** ({cf:.2f})")
            else:
                st.warning("لم يتم اكتشاف أي إشارة.")
    with col2:
        sentence_panel("img")

# ---------------- Video ----------------
with tab_vid:
    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        up = st.file_uploader("ارفع فيديو", type=["mp4", "mov", "avi", "mkv"])
        auto_build = st.checkbox("🧠 بناء الجملة تلقائياً من الفيديو", value=True)
        if up:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
            tf.write(up.read()); tf.close()
            cap = cv2.VideoCapture(tf.name)
            frame_slot = st.empty()
            status = st.empty()
            stop = st.button("⏹️ إيقاف")
            n = 0
            for result in predict(tf.name, conf=conf, iou=iou, imgsz=imgsz, stream=True):
                if stop:
                    break
                frame_slot.image(result.plot()[:, :, ::-1], channels="RGB")
                if auto_build:
                    label, cf = _top_detection(result)
                    added = builder.observe(label, cf) if label else None
                    if added:
                        status.success(f"➕ {added}")
                n += 1
            cap.release()
            st.success(f"✅ تم معالجة {n} فريم")
    with col2:
        sentence_panel("vid")

# ---------------- About ----------------
with tab_about:
    st.markdown("""
### 🤟 مترجم لغة الإشارة العربية

تطبيق ذكي يستخدم **YOLOv26** لاكتشاف حروف لغة الإشارة العربية في الوقت الفعلي،
ثم يبنيها تلقائياً لكلمات وجمل، وينطقها ويترجمها لأي لغة.

**المميزات:**
- 🎯 كشف فوري للحروف بدقة عالية عبر WebRTC
- ✍️ بناء تلقائي للكلمات والجمل مع تصفية ذكية للضوضاء
- 🔊 نطق النص بالعربي والإنجليزي (Text-to-Speech)
- 🌍 ترجمة لـ 8 لغات
- 🌙 وضع ليلي كامل
- 📱 واجهة عصرية Responsive
- 💾 تصدير الجمل كملفات نصية
- ⚙️ تحكم كامل في حساسية الكشف

**التقنيات:** YOLOv26 · Streamlit · WebRTC · OpenCV · gTTS · deep-translator
""")
