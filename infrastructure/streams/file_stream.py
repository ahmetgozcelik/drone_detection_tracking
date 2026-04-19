"""
file_stream.py — Kayıtlı video dosyası için IStream implementasyonu.

CURSOR İÇİN BAĞLAM:
    MP4/AVI/MKV gibi yerel dosyaları okur.
    is_live=False — StreamManager reconnect denemez, dosya bitince döngü durur.
    Akademik test ve CVPR veri seti analizi için kullanılır (plan.txt Aşama 6).

Kullanım:
    stream = FileStream("/path/to/drone_video.mp4")
    stream.open()
    while stream.is_open():
        frame = stream.read()   # None → dosya bitti
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from core.interfaces.istream import IStream, StreamInfo
from utils.logger import get_logger

log = get_logger(__name__)


class FileStream(IStream):
    """
    Yerel video dosyası kaynağı.

    Args:
        path: Video dosyasının tam yolu (str veya Path).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        self._cap: cv2.VideoCapture | None = None
        self._total_frames: int = 0
        self._eof: bool = False          # Dosya bitti mi?
        self._frame_delay: float = 0.0   # FPS'e göre kare arası bekleme (saniye)
        self._last_read_time: float = 0.0

    def open(self) -> bool:
        if not self._path.is_file():
            log.error("Video dosyası bulunamadı: %s", self._path)
            return False

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            log.error("Video dosyası açılamadı: %s", self._path)
            self._cap = None
            return False

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        # Geçersiz FPS değerlerini düzelt (bazı codec'ler 0 döndürür)
        if source_fps <= 0 or source_fps > 120:
            source_fps = 30.0
        self._frame_delay = 1.0 / source_fps
        self._eof = False
        self._last_read_time = 0.0
        log.info(
            "Video dosyası açıldı: %s | Toplam kare: %d | FPS: %.1f",
            self._path.name,
            self._total_frames,
            source_fps,
        )
        return True

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened() or self._eof:
            return None

        # FPS throttle — videonun gerçek hızında oku
        now = time.monotonic()
        elapsed = now - self._last_read_time
        if self._last_read_time > 0 and elapsed < self._frame_delay:
            time.sleep(self._frame_delay - elapsed)
        self._last_read_time = time.monotonic()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            log.info("Video dosyası sona erdi: %s", self._path.name)
            self._eof = True
            return None
        return frame

    def is_open(self) -> bool:
        if self._cap is None or not self._cap.isOpened():
            return False
        return not self._eof

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Video dosyası serbest bırakıldı: %s", self._path.name)

    def get_info(self) -> StreamInfo:
        w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if self._cap else 0
        h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0
        fps = self._cap.get(cv2.CAP_PROP_FPS)               if self._cap else 0.0
        return StreamInfo(
            source_uri=str(self._path),
            width=w,
            height=h,
            fps=fps,
            is_live=False,
        )