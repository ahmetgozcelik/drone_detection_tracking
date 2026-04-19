"""
ITracker — Nesne takip algoritmaları için temel arayüz (SOLID: Interface Segregation)

CURSOR İÇİN BAĞLAM:
    CSRT (hassas) ve KCF (hızlı) takipçileri bu arayüzü implemente eder.
    hybrid_tracker.py bu tipi kullanır; hangi OpenCV tracker'ın seçildiğini bilmez.
    Yeni bir takip algoritması eklemek istersen sadece bu arayüzü implemente et.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackResult:
    """Tek bir takip adımının sonucu."""
    success: bool                        # Takipçi hedefi bulabildi mi?
    bbox: tuple[int, int, int, int]      # (x, y, w, h) — tespit başarısızsa (0,0,0,0)
    confidence: float = 1.0              # CSRT güven skoru döndürür; KCF döndürmez (varsayılan 1.0)


class ITracker(ABC):
    """
    Tüm OpenCV tabanlı takip algoritmaları için soyut temel sınıf.

    Kullanım akışı (hybrid_tracker.py içinde):
        tracker = CsrtTracker()
        tracker.init(frame, bbox)         # YOLO tespitinden gelen bbox ile başlat
        result = tracker.update(frame)    # Her yeni karede çağır
        if not result.success:
            # Takip kaybedildi → IDetector'a geri dön
    """

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
        """
        Takipçiyi verilen ilk konumla başlat.

        Args:
            frame: BGR formatında ilk kare.
            bbox: Takip başlangıç noktası (x, y, w, h).
                  IDetector.detect() çıktısının Detection.bbox'ı ile aynı formattır.

        Returns:
            True: Takipçi başarıyla başlatıldı.
            False: Başlatma başarısız (geçersiz bbox veya kare boyutu).
        """
        ...

    @abstractmethod
    def update(self, frame: np.ndarray) -> TrackResult:
        """
        Yeni karede takipçiyi güncelle ve yeni konumu döndür.

        Args:
            frame: BGR formatında güncel kare.

        Returns:
            TrackResult — success=False ise hybrid_tracker.py algılamaya geri döner.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        İç durumu temizle. Takip kaybedilince veya yeni hedef gelince çağrılır.
        Bellek sızıntısını önlemek için OpenCV tracker nesnesini yeniden oluşturur.
        """
        ...
