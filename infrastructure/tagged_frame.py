"""
tagged_frame.py — Kamera meta verisi taşıyan numpy dizi sarmalayıcısı.

CURSOR İÇİN BAĞLAM:
    TaggedFrame, normal bir numpy dizisidir (np.ndarray alt sınıfı).
    Tüm OpenCV ve NumPy işlemleri şeffaf biçimde çalışır.
    CompositeStream.read() çıktısına camera_id eklemek için kullanılır.
    pipeline.py bu sınıfı doğrudan bilmez; sadece np.ndarray gibi işler.
    SectorManager, frame.meta["camera_id"] ile kamera kimliğini okur.

Kullanım:
    tagged = TaggedFrame.from_frame(bgr_frame, camera_id=1)
    cam_id = tagged.meta.get("camera_id", 0)
"""

from __future__ import annotations

import numpy as np


class TaggedFrame(np.ndarray):
    """
    Numpy dizisi alt sınıfı; meta sözlüğüyle kamera kimliği gibi ek
    bilgileri taşır. Tüm boyutsal/aritmetik/CV işlemler normal çalışır.

    Attributes:
        meta: Kare başına meta veriler. Standart anahtar: "camera_id" (int).
    """

    meta: dict

    def __new__(
        cls,
        input_array: np.ndarray,
        meta: dict | None = None,
    ) -> "TaggedFrame":
        obj = np.asarray(input_array).view(cls)
        obj.meta = meta.copy() if meta else {}
        return obj

    def __array_finalize__(self, obj: object) -> None:
        if obj is None:
            return
        # Dilimleme veya kopyalama sırasında meta'yı koru
        self.meta = getattr(obj, "meta", {}).copy()

    @classmethod
    def from_frame(cls, frame: np.ndarray, camera_id: int) -> "TaggedFrame":
        """
        Varolan bir BGR frame'i verilen camera_id ile etiketler.

        Args:
            frame:     Kaynak BGR numpy dizisi (H × W × 3).
            camera_id: Kamera kimliği (0 veya 1).

        Returns:
            camera_id bilgisi taşıyan TaggedFrame.
        """
        return cls(frame, meta={"camera_id": camera_id})
