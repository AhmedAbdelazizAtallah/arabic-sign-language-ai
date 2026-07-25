"""ترجمة ونطق النصوص."""
from __future__ import annotations

import io
from functools import lru_cache


@lru_cache(maxsize=256)
def translate_ar_to_en(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="ar", target="en").translate(text)
    except Exception as e:  # noqa: BLE001
        return f"[translation error: {e}]"


@lru_cache(maxsize=256)
def translate(text: str, target: str = "en", source: str = "ar") -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:  # noqa: BLE001
        return f"[translation error: {e}]"


def tts_bytes(text: str, lang: str = "ar") -> bytes:
    """يرجّع MP3 bytes لنطق النص."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    return buf.getvalue()