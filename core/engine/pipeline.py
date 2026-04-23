"""
pipeline.py — Capture + Inference thread'leri; SystemController ve metrikler.
"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable
from queue import Empty, Queue

import numpy as np

from configs import settings
from configs.constants import (
    INFERENCE_THREAD_NAME,
    QUEUE_MAXSIZE,
    STATUS_DETECT,
    STATUS_LOST,
    STATUS_TRACK,
)
from core.engine.controller import SystemController
from core.interfaces import IStream
from core.trackers.hybrid_tracker import HybridFrameResult
from infrastructure.stream_manager import StreamManager
from utils.logger import get_logger
from utils.metrics import FpsMeter, LatencyTimer, MetricsSnapshot, PerformanceMonitor

log = get_logger(__name__)


def _snapshot_mode(status: str) -> str:
    """HybridFrameResult.status → MetricsSnapshot.mode"""
    return {
        STATUS_DETECT: "DETECT",
        STATUS_TRACK:  "TRACK",
        STATUS_LOST:   "LOST",
    }.get(status, "IDLE")


class Pipeline:
    """
    Thread 1: StreamManager (capture).
    Thread 2: Kuyruk → SystemController.process → metrikler → on_result callback.

    Args:
        stream:     IStream implementasyonu.
        on_result:  Callable[[np.ndarray, HybridFrameResult, MetricsSnapshot], None]
                    — frame, sonuç ve metriklerle UI'yı günceller.
    """

    def __init__(
        self,
        stream: IStream,
        on_result: Callable[[np.ndarray, HybridFrameResult, MetricsSnapshot], None],
    ) -> None:
        self._stream = stream
        self._on_result = on_result
        self._queue: Queue[np.ndarray] = Queue(maxsize=QUEUE_MAXSIZE)
        self._stream_manager: StreamManager | None = None
        self._controller: SystemController | None = None
        self._inference_stop = threading.Event()
        self._inference_thread: threading.Thread | None = None
        self._started = False
        # Bir önceki kare gecikmesi (EWM) — ileri kestirim için sonraki kareye taşınır
        self._ewm_latency_ms: float = 0.0

        perf = settings.get("performance", {})
        self._gc_interval_frames: int = int(perf.get("gc_interval_frames", 100))
        self._inference_join_timeout_sec: float = float(
            perf.get("inference_join_timeout_sec", 10.0),
        )

        self._fps_meter    = FpsMeter(window=30)
        self._latency_timer = LatencyTimer()
        self._perf_monitor  = PerformanceMonitor()

    def start(self) -> None:
        if self._started:
            log.warning("Pipeline zaten başlatılmış.")
            return

        self._ewm_latency_ms = 0.0
        self._controller = SystemController.build_default()
        self._controller.load()

        self._stream_manager = StreamManager(self._stream, self._queue)
        self._inference_stop.clear()

        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name=INFERENCE_THREAD_NAME,
            daemon=True,
        )
        self._stream_manager.start()
        self._inference_thread.start()
        self._started = True
        log.info("Pipeline başlatıldı (Capture + Inference).")

    def stop(self) -> None:
        def _drain_frame_queue(phase: str) -> int:
            n = 0
            while True:
                try:
                    self._queue.get_nowait()
                    n += 1
                except Empty:
                    break
            if n:
                log.debug("Kare kuyruğu boşaltıldı (%s): %d öğe", phase, n)
            return n

        try:
            self._inference_stop.set()
            # Önce üreticiyi durdur; kuyruğa yeni kare girmesin
            if self._stream_manager is not None:
                self._stream_manager.stop()
                self._stream_manager = None
            _drain_frame_queue("üretici durdu")
            if self._inference_thread is not None and self._inference_thread.is_alive():
                self._inference_thread.join(timeout=self._inference_join_timeout_sec)
                if self._inference_thread.is_alive():
                    log.warning("Inference thread zaman aşımında sonlandı.")
        finally:
            try:
                with self._queue.mutex:
                    self._queue.queue.clear()  # type: ignore[attr-defined]
            except Exception as e:
                log.debug("Kuyruk içi clear atlandı: %s", e)
            # Tüm asılı kareleri serbest bırak (deadlock / bellek)
            _drain_frame_queue("graceful: finally")
            if self._controller is not None:
                try:
                    self._controller.release()
                except Exception as e:
                    log.exception("SystemController.release hatası: %s", e)
                self._controller = None
            self._inference_thread = None
            self._started = False
            gc.collect()
            log.info("Pipeline durduruldu.")

    def _inference_loop(self) -> None:
        processed = 0
        assert self._controller is not None
        ewm_alpha = 0.1  # yeni örnek ağırlığı; ani sıçramaları yumuşat

        while not self._inference_stop.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except Empty:
                continue

            self._latency_timer.start()
            use_latency = self._ewm_latency_ms
            try:
                result = self._controller.process(frame, latency_ms=use_latency)
                # TaggedFrame ise camera_id'yi result'a aktar (CompositeStream desteği)
                if hasattr(frame, "meta"):
                    result.camera_id = frame.meta.get("camera_id", 0)
            except Exception as e:
                log.exception("SystemController.process hatası: %s", e)
                self._latency_timer.stop()
                continue
            latency_ms = self._latency_timer.stop()
            self._ewm_latency_ms = (1.0 - ewm_alpha) * self._ewm_latency_ms + ewm_alpha * float(
                latency_ms,
            )

            self._fps_meter.tick()
            processed += 1
            snapshot = self._perf_monitor.snapshot(
                fps=self._fps_meter.fps,
                latency_ms=latency_ms,
                mode=_snapshot_mode(result.status),
            )

            if self._gc_interval_frames > 0 and processed % self._gc_interval_frames == 0:
                gc.collect()

            try:
                self._on_result(frame, result, snapshot)
            except Exception as e:
                log.exception("on_result callback hatası: %s", e)

        log.debug("Inference döngüsü sonlandı.")
