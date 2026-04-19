"""
metrics.py — FPS, Inference Latency ve RAM ölçüm araçları.

CURSOR İÇİN BAĞLAM:
    pipeline.py her inference döngüsünde bu sınıfları kullanır.
    UI (viewmodel) MetricsSnapshot'u alıp ekrana basar.
    PerformanceMonitor RAM eşiğini aşınca GC tetikler (target_hardware.txt: 2GB limit).
    Bu dosya hiçbir core/ui sınıfına bağımlı değildir — bağımlılık yönü tek yönlüdür.

Kullanım (pipeline.py içinde):
    fps_meter   = FpsMeter(window=30)
    latency     = LatencyTimer()
    ram_monitor = PerformanceMonitor()

    latency.start()
    result = detector.detect(frame)
    ms = latency.stop()

    fps_meter.tick()
    snapshot = MetricsSnapshot(fps=fps_meter.fps, latency_ms=ms)

    if ram_monitor.should_gc():
        import gc; gc.collect()
"""

import gc
import time
from collections import deque
from dataclasses import dataclass, field

import psutil

from configs import settings
from configs.constants import FPS_LABEL, LATENCY_LABEL


# ── Veri Nesnesi ─────────────────────────────────────────────────────────────

@dataclass
class MetricsSnapshot:
    """Bir anın anlık performans görüntüsü. UI bunu alıp gösterir."""
    fps: float = 0.0
    latency_ms: float = 0.0
    ram_used_mb: float = 0.0
    mode: str = "IDLE"          # "DETECT" | "TRACK" | "LOST" | "IDLE"

    @property
    def fps_label(self) -> str:
        return FPS_LABEL.format(self.fps)

    @property
    def latency_label(self) -> str:
        return LATENCY_LABEL.format(self.latency_ms)


# ── FPS Ölçer ────────────────────────────────────────────────────────────────

class FpsMeter:
    """
    Kayan pencere ortalamasıyla FPS hesaplar.
    Her kare işlendiğinde tick() çağrılır.

    Args:
        window: Kaç kare üzerinden ortalama alınsın (varsayılan 30).
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Yeni kare geldiğinde çağır."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Mevcut FPS değeri. Yeterli veri yoksa 0.0 döner."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()


# ── Latency Zamanlayıcı ──────────────────────────────────────────────────────

class LatencyTimer:
    """
    Inference süresi (ms) ölçer.
    start() → [model çalışır] → stop() → ms döner.
    """

    def __init__(self) -> None:
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        """Geçen süreyi milisaniye cinsinden döndür."""
        return (time.perf_counter() - self._start) * 1000.0


# ── RAM ve GC Yöneticisi ─────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    RAM kullanımını izler; eşik aşılınca GC tetiklemesi için sinyal verir.
    target_hardware.txt: 2 GB RAM limiti → settings'te max_ram_mb = 1800 (güvenli marj).

    Kullanım:
        monitor = PerformanceMonitor()
        ...
        if monitor.should_gc(frame_count):
            gc.collect()
            log.debug("GC tetiklendi. RAM: %.0f MB", monitor.ram_mb)
    """

    def __init__(self) -> None:
        perf_cfg = settings.get("performance", {})
        self._max_ram_mb: float = perf_cfg.get("max_ram_mb", 1800)
        self._gc_interval: int  = perf_cfg.get("gc_interval_frames", 100)
        self._process = psutil.Process()

    @property
    def ram_mb(self) -> float:
        """Anlık RAM kullanımı (MB)."""
        return self._process.memory_info().rss / (1024 ** 2)

    def should_gc(self, frame_count: int) -> bool:
        """
        GC çalıştırılmalı mı?
        İki koşuldan biri sağlanırsa True döner:
          1. frame_count gc_interval'ın katıysa (periyodik)
          2. RAM kullanımı eşiği aştıysa (acil)
        """
        periodic  = (frame_count % self._gc_interval == 0)
        over_ram  = self.ram_mb > self._max_ram_mb
        return periodic or over_ram

    def snapshot(self, fps: float, latency_ms: float, mode: str) -> MetricsSnapshot:
        """Anlık tüm metrikleri tek nesnede topla. pipeline.py bunu UI'ya gönderir."""
        return MetricsSnapshot(
            fps=fps,
            latency_ms=latency_ms,
            ram_used_mb=self.ram_mb,
            mode=mode,
        )
