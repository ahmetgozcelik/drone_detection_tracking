"""
usb_stream.py — USB / dahili kamera için IStream implementasyonu.

CURSOR İÇİN BAĞLAM:
    cv2.VideoCapture(device_index) üzerinden çalışır.
    StreamManager bu sınıfı Capture Thread içinde kullanır.
    Reconnect mantığı StreamManager'dadır; bu sınıf tek bağlantı döngüsünü yönetir.

Kullanım:
    stream = UsbStream(device_index=0)
    stream.open()
    frame = stream.read()
    stream.release()
"""

from __future__ import annotations

import cv2
import numpy as np

from core.interfaces.istream import IStream, StreamInfo
from utils.logger import get_logger

log = get_logger(__name__)


class UsbStream(IStream):
    """
    USB veya dahili kamera kaynağı.

    Args:
        device_index: Kamera indeksi (0 = /dev/video0, varsayılan).
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            log.error("USB kamera açılamadı. device_index=%d", self._device_index)
            self._cap = None
            return False
        log.info("USB kamera açıldı. device_index=%d", self._device_index)
        return True

    def read(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            log.warning("USB kameradan kare alınamadı.")
            return None
        return frame

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("USB kamera serbest bırakıldı. device_index=%d", self._device_index)

    def get_info(self) -> StreamInfo:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if self._cap else 0
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0
        fps = self._cap.get(cv2.CAP_PROP_FPS)             if self._cap else 0.0
        return StreamInfo(
            source_uri=f"usb://{self._device_index}",
            width=w,
            height=h,
            fps=fps,
            is_live=True,
        )
