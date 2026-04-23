"""
base_trackers.py — OpenCV CSRT ve KCF takipçileri için ITracker sarmalayıcıları.

CURSOR İÇİN BAĞLAM:
    Bu dosya ITracker arayüzünü implemente eder.
    hybrid_tracker.py bu sınıfları ITracker tipiyle kullanır; CSRT mi KCF mi bilmez.
    Sınıf seçimi settings.yaml → tracker.mode ile yapılır.
    Yeni bir OpenCV tracker eklemek için bu dosyaya yeni bir sınıf ekle, başka hiçbir
    dosyaya dokunma.

Kullanım:
    from core.trackers.base_trackers import CsrtTracker, KcfTracker
    tracker: ITracker = CsrtTracker()
    tracker.init(frame, bbox)
    result = tracker.update(frame)
"""

import cv2
import numpy as np

from core.interfaces.itracker import ITracker, TrackResult
from utils.logger import get_logger

log = get_logger(__name__)


class CsrtTracker(ITracker):
    """
    CSRT (Channel and Spatial Reliability Tracking) sarmalayıcısı.

    Özellikler:
        - Yavaş ama hassas (plan.txt: Model 3 - Hassas)
        - Hızlı manevra yapan drone'larda tercih edilir
        - KCF'e göre ~2x daha yüksek CPU maliyeti
        - TrackResult.confidence değeri döndürür (CSRT bunu destekler)

    Edge device notu (target_hardware.txt):
        Jetson Nano'da CSRT ağır gelebilir. settings.yaml → tracker.mode = "kcf"
        ile geçiş yapılabilir; hybrid_tracker.py bunu otomatik yönetir.
    """

    def __init__(self) -> None:
        self._tracker: cv2.TrackerCSRT | None = None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        self._tracker = cv2.TrackerCSRT_create()
        try:
            self._tracker.init(frame, bbox)
            log.debug("CSRT init başarılı. bbox=%s", bbox)
            return True
        except Exception as e:
            log.warning("CSRT init exception. bbox=%s | %s", bbox, e)
            self._tracker = None
            return False

    def update(self, frame: np.ndarray) -> TrackResult:
        if self._tracker is None:
            return TrackResult(success=False, bbox=(0, 0, 0, 0))
        try:
            ok, box = self._tracker.update(frame)
            if not ok or box is None:
                return TrackResult(success=False, bbox=(0, 0, 0, 0))
            x, y, w, h = (int(v) for v in box)
            return TrackResult(success=True, bbox=(x, y, w, h), confidence=1.0)
        except Exception as e:
            log.warning("CSRT update exception: %s", e)
            return TrackResult(success=False, bbox=(0, 0, 0, 0))

    def reset(self) -> None:
        self._tracker = None
        log.debug("CSRT sıfırlandı.")


class KcfTracker(ITracker):
    """
    KCF (Kernelized Correlation Filters) sarmalayıcısı.

    Özellikler:
        - Hızlı ama daha az hassas (plan.txt: Model 2 - Hızlı)
        - Sabit hızlı, düz uçan drone'larda yeterli
        - CSRT'ye göre ~2x daha düşük CPU maliyeti
        - confidence skoru döndürmez (sabit 1.0)

    Edge device notu (target_hardware.txt):
        Raspberry Pi 4'te önerilen mod. Düşük RAM + CPU altında stabil çalışır.
    """

    def __init__(self) -> None:
        self._tracker: cv2.TrackerKCF | None = None

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        self._tracker = cv2.TrackerKCF_create()
        try:
            self._tracker.init(frame, bbox)
            log.debug("KCF init başarılı. bbox=%s", bbox)
            return True
        except Exception as e:
            log.warning("KCF init exception. bbox=%s | %s", bbox, e)
            self._tracker = None
            return False

    def update(self, frame: np.ndarray) -> TrackResult:
        if self._tracker is None:
            return TrackResult(success=False, bbox=(0, 0, 0, 0))
        try:
            ok, box = self._tracker.update(frame)
            if not ok or box is None:
                return TrackResult(success=False, bbox=(0, 0, 0, 0))
            x, y, w, h = (int(v) for v in box)
            return TrackResult(success=True, bbox=(x, y, w, h), confidence=1.0)
        except Exception as e:
            log.warning("KCF update exception: %s", e)
            return TrackResult(success=False, bbox=(0, 0, 0, 0))

    def reset(self) -> None:
        self._tracker = None
        log.debug("KCF sıfırlandı.")


def create_tracker(mode: str) -> ITracker:
    """
    settings.yaml → tracker.mode değerine göre doğru takipçiyi oluşturan fabrika fonksiyonu.
    hybrid_tracker.py bu fonksiyonu kullanır; sınıf isimlerini doğrudan bilmez.

    Args:
        mode: "csrt" veya "kcf"

    Returns:
        ITracker implementasyonu.

    Raises:
        ValueError: Geçersiz mod girilirse.
    """
    mode = mode.lower().strip()
    if mode == "csrt":
        log.debug("Tracker oluşturuldu: CSRT")
        return CsrtTracker()
    if mode == "kcf":
        log.debug("Tracker oluşturuldu: KCF")
        return KcfTracker()
    raise ValueError(f"Geçersiz tracker modu: '{mode}'. Geçerli değerler: 'csrt', 'kcf'")