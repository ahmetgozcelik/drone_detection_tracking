"""
controller.py — Strategy Pattern ile dedektör ve hibrit takipçi yönetimi.

CURSOR İÇİN BAĞLAM:
    Bu sınıf tüm core bileşenlerini (IDetector, TrackerPool) bir arada tutar.
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

import numpy as np

from configs import settings
from core.detectors.yolo_onnx import YoloOnnxDetector
from core.interfaces.idetector import IDetector
from core.trackers.hybrid_tracker import HybridFrameResult
from core.trackers.tracker_pool import TrackerPool
from utils.logger import get_logger

log = get_logger(__name__)


class SystemController:
    """
    Dedektör ve hibrit ``TrackerPool`` sarmalaması; pipeline tek giriş noktası.

    Doğrudan ``HybridTracker`` referansı yok: çoklu hedef takibi ``TrackerPool`` ile.

    Args:
        detector: IDetector implementasyonu.
        pool:     İsteğe bağlı dış enjekte ``TrackerPool``; verilmezse aynı dedektörle oluşur.
    """

    def __init__(self, detector: IDetector, pool: TrackerPool | None = None) -> None:
        self._detector = detector
        self._pool: TrackerPool = pool if pool is not None else TrackerPool(detector)
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
        self._pool.release()
        self._detector.release()
        self._loaded = False
        log.info("SystemController serbest bırakıldı.")

    # ── Ana İşlem ────────────────────────────────────────────────────────────

    def process(
        self, frame: np.ndarray, latency_ms: float = 0.0,
    ) -> HybridFrameResult:
        """
        Kareyi işle: YOLO bir kez, ``TrackerPool`` tüm hibrit izleyicileri günceller.

        Args:
            frame: BGR formatında numpy dizisi. StreamManager'dan kuyruk üzerinden gelir.
            latency_ms: Bir önceki kareden (EWM) gelen gecikme; Kalman ileri kestirim.

        Returns:
            ``HybridFrameResult`` (``targets: List[TargetData]``) — UI ve metrik.

        Raises:
            RuntimeError: load() çağrılmadan process() çağrılırsa.
        """
        if not self._loaded:
            raise RuntimeError("SystemController.load() önce çağrılmalı.")
        return self._pool.process(
            frame, camera_id=0, latency_ms=latency_ms,
        )

    # ── Durum Sorgu ──────────────────────────────────────────────────────────

    @property
    def current_state(self) -> str:
        """Havuz özet durumu (çoklu izleyici)."""
        return self._pool.state_summary

    @property
    def is_loaded(self) -> bool:
        return self._loaded
