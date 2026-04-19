"""
IStream — Video akış kaynakları için temel arayüz (SOLID: Interface Segregation)

CURSOR İÇİN BAĞLAM:
    USB kamera, RTSP stream ve dosya okuyucu bu arayüzü implemente eder.
    pipeline.py ve stream_manager.py bu tipi kullanır; kaynağın türünü bilmez.
    Yeni bir kaynak eklemek (örn. termal kamera) için sadece bu arayüzü implemente et.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class StreamInfo:
    """Bağlı akış kaynağının meta verisi."""
    source_uri: str          # "rtsp://...", "/dev/video0", "/path/to/video.mp4"
    width: int
    height: int
    fps: float
    is_live: bool            # True = canlı kamera | False = kayıtlı dosya


class IStream(ABC):
    """
    Tüm video kaynakları için soyut temel sınıf.

    Kullanım akışı (pipeline.py Capture Thread içinde):
        stream = RtspStream("rtsp://192.168.1.1/live")
        stream.open()
        while stream.is_open():
            frame = stream.read()
            if frame is not None:
                queue.put(frame)
        stream.release()

    Not: Reconnect mantığı IStream içinde DEĞİL stream_manager.py içindedir.
         IStream yalnızca tek bir bağlantı yaşam döngüsünü yönetir.
    """

    @abstractmethod
    def open(self) -> bool:
        """
        Akış kaynağını aç ve bağlantıyı kur.

        Returns:
            True: Bağlantı başarılı.
            False: Kaynak bulunamadı veya bağlantı reddedildi.

        Raises:
            ConnectionError: RTSP kimlik doğrulama başarısız olursa.
        """
        ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """
        Akıştan bir sonraki kareyi oku.

        Returns:
            BGR formatında numpy dizisi (H x W x 3).
            None: Kare alınamadı (akış kesildi, dosya sona erdi).
        """
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """
        Akış hâlâ açık ve okunabilir durumda mı?
        Capture Thread döngü koşulu olarak kullanır.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """
        cv2.VideoCapture veya soket bağlantısını kapat, kaynağı serbest bırak.
        Her zaman finally bloğunda çağrılmalıdır.
        """
        ...

    @abstractmethod
    def get_info(self) -> StreamInfo:
        """
        Bağlı kaynak hakkında meta veri döndür.
        UI'daki durum paneli ve logger bu bilgiyi kullanır.
        """
        ...
