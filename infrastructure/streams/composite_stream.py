"""
composite_stream.py — Birden fazla IStream'i tek IStream olarak sunar.

CURSOR İÇİN BAĞLAM:
    CompositeStream, IStream arayüzünü implemente eder.
    Her read() çağrısında sıradaki kameradan frame okur (round-robin).
    Döndürülen frame, TaggedFrame alt sınıfıdır → meta["camera_id"] taşır.
    StreamManager ve Pipeline bu sınıfı sıradan bir IStream gibi kullanır;
    çift kamera yapısını bilmesi gerekmez. (Open/Closed Principle)

Kullanım (main_window.py içinde):
    from infrastructure.streams.composite_stream import CompositeStream
    from infrastructure.streams.usb_stream import UsbStream

    streams = [UsbStream(0), UsbStream(1)]
    stream = CompositeStream(streams)
    pipeline = Pipeline(stream=stream, on_result=...)
"""

from __future__ import annotations

import numpy as np

from core.interfaces.istream import IStream, StreamInfo
from infrastructure.tagged_frame import TaggedFrame
from utils.logger import get_logger

log = get_logger(__name__)


class CompositeStream(IStream):
    """
    Birden fazla IStream kaynağını round-robin sıralamasıyla birleştiren sarmalayıcı.

    Her read() çağrısı:
        1. Sıradaki kamerayı seçer.
        2. O kameradan frame okur.
        3. Frame'i TaggedFrame olarak meta["camera_id"] ile etiketler.
        4. Bir sonraki kameraya geçer.

    Bir kamera None döndürürse sıradaki denenir (tümü None ise None döner).

    Args:
        streams: IStream implementasyonlarının listesi (en az 1 eleman).
    """

    def __init__(self, streams: list[IStream]) -> None:
        if not streams:
            raise ValueError("CompositeStream: en az 1 IStream gerekli.")
        self._streams = streams
        self._current_idx: int = 0

    # ── IStream Arayüzü ───────────────────────────────────────────────────────

    def open(self) -> bool:
        """Tüm alt stream'leri açar. Herhangi biri açılamazsa False döner."""
        results = []
        for i, s in enumerate(self._streams):
            ok = s.open()
            results.append(ok)
            if ok:
                log.info("CompositeStream[%d] açıldı: %s", i, s.__class__.__name__)
            else:
                log.error("CompositeStream[%d] açılamadı: %s", i, s.__class__.__name__)
        return all(results)

    def read(self) -> np.ndarray | None:
        """
        Sıradaki kameradan TaggedFrame döndürür.

        Tüm kameralar None dönerse None döner.
        """
        for _ in range(len(self._streams)):
            idx = self._current_idx
            stream = self._streams[idx]
            self._current_idx = (idx + 1) % len(self._streams)

            frame = stream.read()
            if frame is not None:
                return TaggedFrame.from_frame(frame, camera_id=idx)

        return None

    def is_open(self) -> bool:
        """En az bir alt stream açıksa True."""
        return any(s.is_open() for s in self._streams)

    def release(self) -> None:
        """Tüm alt stream'leri serbest bırakır."""
        for i, s in enumerate(self._streams):
            s.release()
            log.info("CompositeStream[%d] serbest bırakıldı.", i)

    def get_info(self) -> StreamInfo:
        """İlk açık alt stream'in bilgisini döndürür."""
        for s in self._streams:
            if s.is_open():
                info = s.get_info()
                return StreamInfo(
                    source_uri=f"composite://{info.source_uri}+{len(self._streams)}cams",
                    width=info.width,
                    height=info.height,
                    fps=info.fps,
                    is_live=True,
                )
        return StreamInfo(
            source_uri=f"composite://offline+{len(self._streams)}cams",
            width=0,
            height=0,
            fps=0.0,
            is_live=True,
        )
