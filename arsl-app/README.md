# 🤟 Arabic Sign Language Detector (YOLOv8)

تطبيق Python لكشف حروف/كلمات لغة الإشارة العربية باستخدام موديل YOLOv8
مُدرّب مسبقاً (`best.pt`). يشتغل من الكاميرا مباشرة أو على صور/فيديوهات مرفوعة،
ويوفر واجهتين:

- **CLI سريع** (OpenCV window) — للأداء الحقيقي real-time.
- **Streamlit UI** — واجهة ويب محلية لرفع الملفات وتجربة الموديل.

---

## 📁 هيكل المشروع

```
arsl-app/
├── app.py                # واجهة Streamlit
├── requirements.txt
├── models/
│   └── best.pt           # (ضعه يدوياً — مستبعد من git)
├── src/
│   ├── config.py         # إعدادات ومسارات
│   ├── detector.py       # تحميل الموديل + predict (مع warmup و caching)
│   ├── webcam.py         # real-time من الكاميرا
│   └── cli.py            # واجهة سطر الأوامر
└── outputs/              # نتائج الصور/الفيديو
```

---

## ⚙️ التنصيب

```bash
git clone <repo-url>
cd arsl-app

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ضع ملف الموديل:

```
arsl-app/models/best.pt
```

> لو عندك GPU مع CUDA، `ultralytics` هيستخدمها تلقائياً وتحصل على أعلى FPS.

---

## 🚀 الاستخدام

### 1) الكاميرا (أسرع — OpenCV مباشرة)

```bash
python -m src.cli webcam
python -m src.cli webcam --camera 1 --conf 0.4 --imgsz 480
```
اضغط `Q` للخروج.

### 2) صورة

```bash
python -m src.cli image path/to/photo.jpg
```
النتيجة تُحفظ في `outputs/`.

### 3) فيديو

```bash
python -m src.cli video path/to/clip.mp4
```

### 4) واجهة Streamlit (رفع صور/فيديو + snapshot من الكاميرا)

```bash
streamlit run app.py
```

---

## ⚡ تحسينات الأداء المطبّقة

- **Model caching** عبر `lru_cache` — تحميل مرة واحدة فقط.
- **Warmup** بعد التحميل عشان أول frame يكون سريع.
- **Auto device detection** — GPU لو متاح وإلا CPU.
- **Stream mode** للفيديوهات (`stream=True`) عشان الذاكرة تبقى ثابتة.
- **FPS smoothing** (EMA) في الكاميرا.
- خفض `imgsz` (مثلاً 480) بيزوّد الـ FPS بشكل ملحوظ.

---

## 📤 رفع المشروع على GitHub

```bash
cd arsl-app
git init
git add .
git commit -m "Initial commit: Arabic Sign Language Detector"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

> **مهم:** `best.pt` كبير الحجم — لا ترفعه على git مباشرة.
> استخدم **GitHub Release** أو **Git LFS**:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes models/best.pt
> ```

---

## 🧪 اختبار سريع

```bash
python -c "from src.detector import load_model, class_names; load_model(); print(class_names())"
```

---

## 📜 الرخصة

MIT