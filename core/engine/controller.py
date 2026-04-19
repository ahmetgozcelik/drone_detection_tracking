"""
controller.py — Strategy Pattern ile dedektör ve hibrit takipçi yönetimi.

CURSOR İÇİN BAĞLAM:
    Bu sınıf tüm core bileşenlerini (IDetector, HybridTracker) bir arada tutar.
    pipeline.py bu sınıfı oluşturur ve process() metodunu çağırır.
    UI katmanı controller'ı doğrudan kullanmaz; pipeline üzerinden iletişim kurar.
    Farklı bir detector (yolo_v8.py) denemek için sadece bu sınıfa yeni bir
    factory metodu eklenir; pipeline.py ve UI değişmez. (Open/Closed Principle)

Kullanım (pipeline.py içinde):
    controller = SystemController.build_default()
    controller.load()
    result = controller.process(frame)   # HybridFrameResult döner
    controller.release()
"""

from __future__ import annotations

from configs import settings
from configs.constants import TRACKER_CSRT, TRACKER_KCF
from core.detectors.yolo_onnx import YoloOnnxDetector
from core.interfaces.idetector import IDetector
from core.trackers.hybrid_tracker import HybridFrameResult, HybridTracker
from utils.logger import get_logger

import numpy as np

log = get_logger(__name__)


class SystemController:
    """
    Detector ve HybridTracker'ı sarar; pipeline için tek işlem noktası sağlar.

    Args:
        detector: IDetector implementasyonu (varsayılan: YoloOnnxDetector).
    """

    def __init__(self, detector: IDetector) -> None:
        self._detector = detector
        self._hybrid   = HybridTracker(detector)
        self._loaded   = False

    # ── Fabrika ──────────────────────────────────────────────────────────────

    @classmethod
    def build_default(cls) -> "SystemController":
        """
        settings.yaml'a göre varsayılan sistemi oluştur.
        Şu an tek desteklenen detector: YoloOnnxDetector.
        İleride TFLite veya yolo_v8.py eklemek için buraya yeni bir
        classmethod ekle; pipeline.py değişmez.
        """
        log.info("SystemController oluşturuluyor (YoloOnnxDetector).")
        detector = YoloOnnxDetector()
        return cls(detector)

    # ── Yaşam Döngüsü ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Modeli belleğe yükle. Pipeline başlarken bir kez çağrılır."""
        if self._loaded:
            log.warning("SystemController zaten yüklenmiş.")
            return
        self._detector.load()
        self._loaded = True
        log.info(
            "SystemController hazır | tracker.mode=%s | redetect_interval=%s",
            settings["tracker"]["mode"],
            settings["tracker"]["redetect_interval"],
        )

    def release(self) -> None:
        """Kaynakları serbest bırak. Pipeline finally bloğunda çağırır."""
        self._detector.release()
        self._hybrid.reset()
        self._loaded = False
        log.info("SystemController serbest bırakıldı.")

    # ── Ana İşlem ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> HybridFrameResult:
        """
        Tek kareyi işle: YOLO veya tracker — HybridTracker karar verir.

        Args:
            frame: BGR formatında numpy dizisi. StreamManager'dan kuyruk üzerinden gelir.

        Returns:
            HybridFrameResult(bbox, status) — pipeline bunu metrics ve UI'ya iletir.

        Raises:
            RuntimeError: load() çağrılmadan process() çağrılırsa.
        """
        if not self._loaded:
            raise RuntimeError("SystemController.load() önce çağrılmalı.")
        return self._hybrid.process(frame)

    # ── Durum Sorgu ──────────────────────────────────────────────────────────

    @property
    def current_state(self) -> str:
        """HybridTracker'ın mevcut durumu (STATUS_DETECT / TRACK / LOST)."""
        return self._hybrid.state

    @property
    def is_loaded(self) -> bool:
        return self._loaded
