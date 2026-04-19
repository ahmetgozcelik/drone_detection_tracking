"""
stream_manager.py — Video akış kaynağı yönetimi ve otomatik yeniden bağlanma.

CURSOR İÇİN BAĞLAM:
    Bu sınıf IStream arayüzünü KULLANIR, implemente etmez.
    Reconnect mantığı burada; IStream implementasyonları (RtspStream, UsbStream)
    sadece tek bir bağlantı yaşam döngüsünü bilir.
    pipeline.py bu sınıfı Capture Thread içinde kullanır.
    settings.yaml → stream bölümü bu sınıfı konfigüre eder.

Kullanım (pipeline.py Capture Thread içinde):
    manager = StreamManager(stream=RtspStream(url), queue=frame_queue)
    manager.start()     # Thread başlatır
    ...
    manager.stop()      # Güvenli kapatma
"""

import threading
import time
from queue import Full, Queue

from configs import settings
from configs.constants import CAPTURE_THREAD_NAME, QUEUE_MAXSIZE
from core.interfaces import IStream
from utils.logger import get_logger

log = get_logger(__name__)


class StreamManager:
    """
    IStream kaynağını ayrı bir thread'de çalıştırır.
    Bağlantı kopunca settings'teki parametrelerle yeniden bağlanmayı dener.

    Args:
        stream:      IStream implementasyonu (RtspStream, UsbStream, FileStream).
        frame_queue: Inference thread'in okuduğu paylaşımlı kuyruk.
    """

    def __init__(self, stream: IStream, frame_queue: Queue) -> None:
        self._stream       = stream
        self._queue        = frame_queue
        self._stop_event   = threading.Event()
        self._thread       = threading.Thread(
            target=self._run,
            name=CAPTURE_THREAD_NAME,
            daemon=True,          # Ana thread kapanınca bu da kapanır
        )

        cfg = settings.get("stream", {})
        self._reconnect_attempts: int   = cfg.get("reconnect_attempts", 5)
        self._reconnect_delay:    float = cfg.get("reconnect_delay_sec", 2.0)

    # ── Genel API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Capture thread'i başlat."""
        log.info("StreamManager başlatılıyor: %s", self._stream.__class__.__name__)
        self._thread.start()

    def stop(self) -> None:
        """
        Capture thread'i güvenli şekilde durdur.
        pipeline.py finally bloğunda çağırır.
        """
        log.info("StreamManager durduruluyor...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._stream.release()
        log.info("StreamManager durduruldu.")

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    # ── İç Çalışma Mantığı ───────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Capture thread ana döngüsü.
        Bağlantı açılır → kareler kuyruğa atılır → kopunca reconnect denenir.
        """
        if not self._open_with_retry():
            log.error("Akış açılamadı. Capture thread sonlanıyor.")
            return

        info = self._stream.get_info()
        log.info(
            "Akış açıldı | Kaynak: %s | Çözünürlük: %dx%d | FPS: %.1f",
            info.source_uri, info.width, info.height, info.fps,
        )

        while not self._stop_event.is_set():
            frame = self._stream.read()

            if frame is None:
                # Kare alınamadı → dosya bitti mi, yoksa bağlantı mı koptu?
                info = self._stream.get_info()
                if not info.is_live:
                    log.info("Video dosyası tamamlandı. Capture thread sonlanıyor.")
                    break
                log.warning("Akıştan kare alınamadı. Yeniden bağlanılıyor...")
                self._stream.release()

                if not self._open_with_retry():
                    log.error("Yeniden bağlantı başarısız. Capture thread sonlanıyor.")
                    break
                continue

            # Kuyruğa eklemeyi dene; dolu ise eski kareyi at (güncel kare öncelikli)
            # Bu sayede inference thread her zaman en son kareyi işler.
            try:
                self._queue.put_nowait(frame)
            except Full:
                try:
                    self._queue.get_nowait()   # En eski kareyi çıkar
                    self._queue.put_nowait(frame)
                except Exception:
                    pass                        # Yarış durumu; kareyi atla

    def _open_with_retry(self) -> bool:
        """
        Akışı açmayı settings'teki parametrelerle yeniden dene.

        Returns:
            True: Bağlantı başarılı.
            False: Tüm denemeler başarısız.
        """
        for attempt in range(1, self._reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False

            log.info("Bağlantı denemesi %d/%d...", attempt, self._reconnect_attempts)

            if self._stream.open():
                return True

            if attempt < self._reconnect_attempts:
                time.sleep(self._reconnect_delay)

        return False
