"""
sector_manager.py — Çift kamera sektör yönetimi ve pan-tilt servo kontrolü.

CURSOR İÇİN BAĞLAM:
    Bu sınıf Pipeline.on_result callback'i olarak enjekte edilir.
    Her kare sonucunu alır, drone varsa servo konumunu hesaplar ve
    SerialController aracılığıyla ESP32'ye komut gönderir.
    main.py bu sınıfı oluşturur: Pipeline(on_result=sector.handle_frame_result)
    settings.yaml → sectors bölümü konfigürasyon sağlar.

Koordinat Sistemi:
    - Pan:  0° = ön (Kamera A yönü). Pozitif = saat yönü. Aralık: [-60°, +60°]
    - Tilt: 0° = yatay. Pozitif = yukarı. Aralık: [0°, +60°]
    - Azimut: 0° = referans yönü. Pan + montaj açısı + piksel sapması.

Azimut Hesabı:
    pixel_offset_x = (bbox_cx - frame_w/2) / (frame_w/2)   # [-1, 1]
    cam_azimut_offset = pixel_offset_x * (fov_deg / 2)
    world_azimut = pan_deg + mount_angle[camera_id] + cam_azimut_offset

Elevasyon Hesabı:
    pixel_offset_y = (bbox_cy - frame_h/2) / (frame_h/2)   # [-1, 1]
    v_fov = fov_deg * (9/16)                                # 16:9 dikey FOV
    tilt_offset = -pixel_offset_y * (v_fov / 2)
    tilt_cmd = clamp(tilt_center + tilt_offset, tilt_min, tilt_max)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from configs import settings
from configs.constants import STATUS_DETECT, STATUS_LOST, STATUS_TRACK
from core.trackers.hybrid_tracker import HybridFrameResult
from utils.logger import get_logger
from utils.metrics import MetricsSnapshot

if TYPE_CHECKING:
    from infrastructure.serial_controller import SerialController

log = get_logger(__name__)


class SectorManager:
    """
    Çift kameralı 360° kapsama yöneticisi.

    Sorumluluklar:
        1. Kamera bbox piksellerini dünya azimut/elevasyon açısına çevirir.
        2. SerialController aracılığıyla pan-tilt servolarını komuta alır.
        3. Drone bulunamadığında sweep (süpürme) taraması yapar.
        4. Drone tespit edilince TRACK moduna geçip izleme yapar.

    Args:
        serial_controller: SerialController örneği. None ise servo komutları
                           yalnızca loglanır, gerçek UART gönderimi olmaz
                           (test ve simülasyon için kullanışlı).
        on_pan_tilt_update: Opsiyonel callback(pan_deg, tilt_deg, mode).
                            UI'nın servo pozisyonunu göstermesi için kullanılır.
    """

    # Çalışma modu sabitleri
    MODE_SWEEP: str = "SWEEP"
    MODE_TRACK: str = "TRACK"

    def __init__(
        self,
        serial_controller: "SerialController | None" = None,
        on_pan_tilt_update: "Callable[[float, float, str], None] | None" = None,
    ) -> None:
        cfg = settings.get("sectors", {})

        self._camera_count: int = int(cfg.get("camera_count", 2))
        self._fov_deg: float = float(cfg.get("camera_fov_deg", 120.0))

        mount_raw = cfg.get("camera_mount_angles", [0.0, 180.0])
        self._mount_angles: list[float] = [float(a) for a in mount_raw]

        pan_range = cfg.get("pan_range_deg", [-60.0, 60.0])
        self._pan_min: float = float(pan_range[0])
        self._pan_max: float = float(pan_range[1])

        tilt_range = cfg.get("tilt_range_deg", [0.0, 60.0])
        self._tilt_min: float = float(tilt_range[0])
        self._tilt_max: float = float(tilt_range[1])

        self._sweep_enabled: bool = bool(cfg.get("sweep_enabled", True))
        self._sweep_period_sec: float = float(cfg.get("sweep_period_sec", 4.0))

        self._serial = serial_controller
        self._on_pan_tilt_update = on_pan_tilt_update
        self._lock = threading.Lock()

        # Mevcut servo pozisyonları
        self._pan_deg: float = 0.0
        self._tilt_deg: float = (self._tilt_min + self._tilt_max) / 2.0

        # Sweep durum makinesi
        self._mode: str = self.MODE_SWEEP
        self._sweep_dir: float = 1.0         # +1: sağa, -1: sola
        self._last_sweep_time: float = time.monotonic()

        # İzleme kaybı sayacı → sweep'e geri dönmek için
        self._lost_frames: int = 0
        self._lost_threshold: int = 30       # Bu kadar kayıp kare → SWEEP

        # UI için dışarıdan okunabilir durum (kilit gerekmez; okuma race zararsız)
        self._active_camera_id: int = 0
        self._last_azimut: float = 0.0

    # ── Dışa Açık Özellikler ──────────────────────────────────────────────────

    @property
    def pan_deg(self) -> float:
        """Mevcut pan açısı (derece)."""
        return self._pan_deg

    @property
    def tilt_deg(self) -> float:
        """Mevcut tilt açısı (derece)."""
        return self._tilt_deg

    @property
    def mode(self) -> str:
        """Mevcut çalışma modu: SectorManager.MODE_SWEEP veya MODE_TRACK."""
        return self._mode

    @property
    def active_camera_id(self) -> int:
        """Son işlenen karenin kamera kimliği (0 veya 1)."""
        return self._active_camera_id

    @property
    def last_azimut(self) -> float:
        """Son hesaplanan dünya azimutü (derece)."""
        return self._last_azimut

    # ── Pipeline Callback ─────────────────────────────────────────────────────

    def handle_frame_result(
        self,
        frame: np.ndarray,
        result: HybridFrameResult,
        snap: MetricsSnapshot,
    ) -> None:
        """
        Pipeline.on_result callback'i — her kare için inference thread'den çağrılır.

        Kamera kimliğini result.camera_id'den okur (pipeline.py tarafından
        TaggedFrame.meta["camera_id"]'dan çıkarılmış olmalı).

        Args:
            frame:  Kare (gerekirse TaggedFrame; camera_id zaten result'ta).
            result: HybridFrameResult (bbox, status, camera_id).
            snap:   Performans metrikleri snapshot (loglama için).
        """
        camera_id: int = getattr(result, "camera_id", 0)

        with self._lock:
            self._active_camera_id = camera_id

            if result.status == STATUS_TRACK and result.bbox != (0, 0, 0, 0):
                self._handle_track(frame, result, camera_id)
                self._lost_frames = 0

            elif result.status == STATUS_LOST:
                self._lost_frames += 1
                self._maybe_enter_sweep()

            else:
                # DETECTING ya da TRACK fakat geçersiz bbox
                if self._mode == self.MODE_TRACK:
                    self._lost_frames += 1
                    self._maybe_enter_sweep()
                elif self._sweep_enabled:
                    self._step_sweep()

    # ── Koordinat Dönüşümleri ─────────────────────────────────────────────────

    def azimut_from_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_width: int,
        camera_id: int,
    ) -> float:
        """
        Bounding box merkezi + kamera kimliğinden dünya azimutunu hesaplar.

        Formül:
            pixel_offset = (bbox_cx - frame_w/2) / (frame_w/2)
            cam_offset   = pixel_offset * (fov_deg / 2)
            azimut       = pan_deg + mount_angle[camera_id] + cam_offset

        Args:
            bbox:        (x, y, w, h) piksel koordinatları.
            frame_width: Kare genişliği piksel cinsinden.
            camera_id:   Kamera kimliği (0 = ön, 1 = arka).

        Returns:
            Dünya azimutü, -180..180 aralığında normalize edilmiş.
        """
        x, _y, w, _h = bbox
        bbox_cx = x + w / 2.0

        pixel_offset = (bbox_cx - frame_width / 2.0) / (frame_width / 2.0)
        cam_offset = pixel_offset * (self._fov_deg / 2.0)

        mount = self._mount_angles[camera_id] if camera_id < len(self._mount_angles) else 0.0
        raw_azimut = self._pan_deg + mount + cam_offset

        # -180..180 normalleştirme
        return float(((raw_azimut + 180.0) % 360.0) - 180.0)

    def tilt_from_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_height: int,
    ) -> float:
        """
        Bounding box dikey merkezi + kare yüksekliğinden tilt komutunu hesaplar.

        Formül:
            pixel_offset_y = (bbox_cy - frame_h/2) / (frame_h/2)
            v_fov          = fov_deg * (9/16)
            tilt_cmd       = clamp(tilt_center - pixel_offset_y * v_fov/2, min, max)

        Args:
            bbox:         (x, y, w, h) piksel koordinatları.
            frame_height: Kare yüksekliği piksel cinsinden.

        Returns:
            Tilt açısı, [tilt_min, tilt_max] aralığında kırpılmış.
        """
        _x, y, _w, h = bbox
        bbox_cy = y + h / 2.0

        pixel_offset_y = (bbox_cy - frame_height / 2.0) / (frame_height / 2.0)

        # 16:9 aspect ratio varsayımıyla dikey FOV tahmini
        v_fov = self._fov_deg * (9.0 / 16.0)

        tilt_center = (self._tilt_min + self._tilt_max) / 2.0
        # Yukarıda olan nesne (küçük y) → pozitif tilt
        tilt_cmd = tilt_center - pixel_offset_y * (v_fov / 2.0)

        return float(max(self._tilt_min, min(self._tilt_max, tilt_cmd)))

    # ── Durum Makinesi ────────────────────────────────────────────────────────

    def _handle_track(
        self,
        frame: np.ndarray,
        result: HybridFrameResult,
        camera_id: int,
    ) -> None:
        """Drone tespit edildiğinde azimut/tilt hesapla ve servo komutunu gönder."""
        if self._mode != self.MODE_TRACK:
            log.info("SectorManager: SWEEP → TRACK (kamera %d)", camera_id)
            self._mode = self.MODE_TRACK

        h, w = frame.shape[:2]
        azimut = self.azimut_from_bbox(result.bbox, w, camera_id)
        tilt = self.tilt_from_bbox(result.bbox, h)

        # Pan = azimut - montaj açısı → servo referans çerçevesine dönüştür
        mount = self._mount_angles[camera_id] if camera_id < len(self._mount_angles) else 0.0
        pan_cmd = azimut - mount
        pan_cmd = float(max(self._pan_min, min(self._pan_max, pan_cmd)))

        self._pan_deg = pan_cmd
        self._tilt_deg = tilt
        self._last_azimut = azimut

        self._send_servo(pan_cmd, tilt)
        log.debug(
            "TRACK cam=%d  azimut=%.1f°  pan=%.1f°  tilt=%.1f°",
            camera_id, azimut, pan_cmd, tilt,
        )

    def _maybe_enter_sweep(self) -> None:
        """Kayıp kare sayacı eşiği geçince sweep moduna geçer."""
        if self._lost_frames >= self._lost_threshold:
            if self._mode != self.MODE_SWEEP:
                log.info(
                    "SectorManager: TRACK → SWEEP (%d kayıp kare)", self._lost_frames
                )
                self._mode = self.MODE_SWEEP
                self._lost_frames = 0
            if self._sweep_enabled:
                self._step_sweep()

    def _step_sweep(self) -> None:
        """
        Tek sweep adımı: pan servoyu süpürme hareketi ile yavaşça ileri-geri çalıştırır.

        sweep_period_sec'in yarısı tek yön için ayrılır:
            hız = (pan_max - pan_min) / (sweep_period_sec / 2)
        """
        now = time.monotonic()
        elapsed = now - self._last_sweep_time
        self._last_sweep_time = now

        half_period = max(self._sweep_period_sec / 2.0, 0.01)
        pan_range = self._pan_max - self._pan_min
        step = (elapsed / half_period) * pan_range * self._sweep_dir

        new_pan = self._pan_deg + step

        if new_pan >= self._pan_max:
            new_pan = self._pan_max
            self._sweep_dir = -1.0
        elif new_pan <= self._pan_min:
            new_pan = self._pan_min
            self._sweep_dir = 1.0

        self._pan_deg = new_pan
        self._send_servo(new_pan, self._tilt_deg)

    # ── Servo Gönderimi ───────────────────────────────────────────────────────

    def _send_servo(self, pan_deg: float, tilt_deg: float) -> None:
        """SerialController varsa komutu gönderir; yoksa yalnızca debug loglar."""
        if self._serial is not None and self._serial.is_connected:
            self._serial.send_pan_tilt(pan_deg, tilt_deg)
        else:
            log.debug(
                "Servo komut (seri port yok): PAN=%.1f°  TILT=%.1f°",
                pan_deg, tilt_deg,
            )

        if self._on_pan_tilt_update is not None:
            try:
                self._on_pan_tilt_update(pan_deg, tilt_deg, self._mode)
            except Exception as exc:
                log.debug("on_pan_tilt_update callback hatası: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SectorManager("
            f"cameras={self._camera_count}, "
            f"fov={self._fov_deg}°, "
            f"mode={self._mode}, "
            f"pan={self._pan_deg:.1f}°, "
            f"tilt={self._tilt_deg:.1f}°"
            f")"
        )
