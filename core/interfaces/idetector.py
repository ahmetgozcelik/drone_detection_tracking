"""
IDetector — Algılama motoru için temel arayüz (SOLID: Interface Segregation)

CURSOR İÇİN BAĞLAM:
    Bu arayüzü implemente eden her sınıf (yolo_onnx.py, yolo_v8.py vb.)
    bu 3 metodu sağlamak ZORUNDADIR. Arayüzü değiştirme; sadece implemente et.
    engine/controller.py bu tipi beklediğinden tip uyumsuzluğu runtime hatası verir.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    """Tek bir tespit sonucunu taşıyan veri nesnesi."""
    bbox: tuple[int, int, int, int]   # (x, y, w, h) — OpenCV tracker formatıyla uyumlu
    confidence: float                  # 0.0 – 1.0 arası güven skoru
    class_id: int                      # Sınıf indeksi (bu projede daima 0 = drone)
    label: str = "drone"


class IDetector(ABC):
    """
    Tüm algılama motorları için soyut temel sınıf.

    Kullanım akışı:
        detector = YoloOnnxDetector(model_path, conf_threshold)
        detector.load()
        detections = detector.detect(frame)
        detector.release()
    """

    @abstractmethod
    def load(self) -> None:
        """
        Model ağırlıklarını ve çıkarım oturumunu belleğe yükle.
        Bu metot yalnızca bir kez çağrılmalı; pipeline başlatılırken tetiklenir.

        Raises:
            FileNotFoundError: Model dosyası bulunamazsa.
            RuntimeError: ONNX oturumu açılamazsa.
        """
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Verilen karedeki drone nesnelerini tespit et.

        Args:
            frame: BGR formatında Ham OpenCV karesi (H x W x 3).

        Returns:
            Detection listesi. Hiçbir şey bulunmazsa boş liste döner.
            Güven eşiği filtrelemesi bu metot içinde yapılır.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        Model oturumunu kapat ve belleği serbest bırak (ONNX session, PyTorch model vb.).
        pipeline.py __del__ veya finally bloğunda çağırır.
        """
        ...
