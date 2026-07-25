"""بناء الكلمات والجمل من الحروف المتتالية المكتشفة."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List, Optional

# خرائط اختيارية لحروف تحكم (لو الموديل بيتوقعها)
CONTROL_LABELS = {
    "space": " ",
    "Space": " ",
    "SPACE": " ",
    "del": "<DEL>",
    "Del": "<DEL>",
    "delete": "<DEL>",
    "nothing": None,
    "none": None,
    "background": None,
}


@dataclass
class SentenceBuilder:
    """يجمّع الحروف المتتالية لكلمة وجملة مع تصفية الضوضاء.

    - stability_frames: عدد الفريمات المتتالية اللازمة لتثبيت الحرف
    - min_confidence: أقل ثقة لقبول التنبؤ
    - cooldown_frames: عدد الفريمات بعد الإضافة قبل قبول حرف جديد
    """

    stability_frames: int = 6
    min_confidence: float = 0.55
    cooldown_frames: int = 4
    history_size: int = 30

    text: str = ""
    last_committed: Optional[str] = None
    _buffer: Deque[str] = field(default_factory=lambda: deque(maxlen=30))
    _cooldown: int = 0
    _history: List[dict] = field(default_factory=list)

    def observe(self, label: Optional[str], confidence: float) -> Optional[str]:
        """يستقبل تنبؤ فريم واحد ويرجّع الحرف المضاف (لو فيه)."""
        if self._cooldown > 0:
            self._cooldown -= 1

        if label is None or confidence < self.min_confidence:
            self._buffer.append("__none__")
            return None

        self._buffer.append(label)

        # لازم آخر stability_frames كلهم نفس الحرف
        if len(self._buffer) < self.stability_frames:
            return None
        window = list(self._buffer)[-self.stability_frames :]
        if len(set(window)) != 1:
            return None

        stable = window[0]
        if self._cooldown > 0:
            return None

        # منع تكرار نفس الحرف مباشرة
        if stable == self.last_committed:
            return None

        char = self._resolve(stable)
        if char is None:
            return None
        if char == "<DEL>":
            self.backspace()
        else:
            self.text += char
            self._history.append({"label": stable, "conf": confidence, "char": char})
            if len(self._history) > self.history_size:
                self._history = self._history[-self.history_size :]

        self.last_committed = stable
        self._cooldown = self.cooldown_frames
        return char

    def _resolve(self, label: str) -> Optional[str]:
        if label in CONTROL_LABELS:
            return CONTROL_LABELS[label]
        return label

    # أوامر يدوية
    def add_space(self) -> None:
        if not self.text.endswith(" "):
            self.text += " "
        self.last_committed = None

    def backspace(self) -> None:
        self.text = self.text[:-1]
        self.last_committed = None

    def clear(self) -> None:
        self.text = ""
        self.last_committed = None
        self._buffer.clear()
        self._history.clear()

    @property
    def words(self) -> List[str]:
        return [w for w in self.text.strip().split(" ") if w]

    @property
    def history(self) -> List[dict]:
        return list(self._history)