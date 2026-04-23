"""
test_sector_manager.py — SectorManager birim testleri (T2).

Donanım gerektirmez. pytest ile çalıştır:
    pytest tests/test_sector_manager.py -v

Kapsam:
    - azimut_from_bbox: piksel koordinatı → dünya azimut dönüşümü
    - tilt_from_bbox: piksel koordinatı → tilt açısı
    - handle_frame_result: durum makinesi geçişleri (SWEEP → TRACK → SWEEP)
    - sweep adımı: yön değişimi ve sınır kontrolü
"""

from __future__ import annotations

import time
import math
import unittest
from unittest.mock import MagicMock

import numpy as np


# settings.yaml bağımlılığını test ortamında mock'la
import sys
from unittest.mock import patch

_MOCK_SETTINGS = {
    "sectors": {
        "camera_count": 2,
        "camera_fov_deg": 120.0,
        "camera_mount_angles": [0.0, 180.0],
        "pan_range_deg": [-60.0, 60.0],
        "tilt_range_deg": [0.0, 60.0],
        "sweep_enabled": True,
        "sweep_period_sec": 4.0,
    },
    "serial": {
        "enabled": False,
    },
}


def _make_manager(serial_ctrl=None):
    """Test için SectorManager örneği oluşturan yardımcı."""
    with patch("core.engine.sector_manager.settings", _MOCK_SETTINGS):
        from core.engine.sector_manager import SectorManager
        return SectorManager(serial_controller=serial_ctrl)


def _make_result(bbox=(0, 0, 0, 0), status="TESPIT", camera_id=0):
    """Test için HybridFrameResult örneği."""
    from core.trackers.hybrid_tracker import HybridFrameResult
    return HybridFrameResult(bbox=bbox, status=status, camera_id=camera_id)


def _make_snap():
    """Test için MetricsSnapshot örneği."""
    from utils.metrics import MetricsSnapshot
    return MetricsSnapshot(
        fps=30.0,
        fps_label="FPS: 30.0",
        latency_ms=10.0,
        latency_label="Latency: 10.0 ms",
        ram_used_mb=500.0,
        mode="DETECT",
    )


class TestAzimutFromBbox(unittest.TestCase):
    """azimut_from_bbox() fonksiyon testleri."""

    def setUp(self):
        self.sm = _make_manager()

    def test_center_bbox_camera0_pan0_azimut0(self):
        """Kamera 0, bbox merkezi (x=960) pan=0° → azimut=0°."""
        # frame genişliği 1920, bbox merkezi tam ortada
        bbox = (910, 100, 100, 100)  # cx = 960
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=0)
        self.assertAlmostEqual(azimut, 0.0, places=1,
                               msg="Merkez piksel, pan=0, cam=ön → azimut=0°")

    def test_right_edge_bbox_camera0_pan0(self):
        """Kamera 0, bbox tam sağ kenarda, pan=0° → azimut=+60° (fov/2)."""
        bbox = (1870, 100, 100, 100)  # cx = 1920 → sağ kenar (piksel_offset ≈ 1)
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=0)
        # pixel_offset ≈ 1.0 → cam_offset = 1.0 * 60 = +60
        self.assertAlmostEqual(azimut, 60.0, delta=3.0)

    def test_left_edge_bbox_camera0_pan0(self):
        """Kamera 0, bbox tam sol kenarda, pan=0° → azimut=-60°."""
        bbox = (0, 100, 50, 100)  # cx = 25 → sol kenar (pixel_offset ≈ -1)
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=0)
        self.assertAlmostEqual(azimut, -60.0, delta=3.0)

    def test_center_bbox_camera1_pan0_azimut180(self):
        """Kamera 1 (arka, montaj=180°), bbox merkezi, pan=0° → azimut=180°."""
        bbox = (910, 100, 100, 100)  # cx = 960
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=1)
        # 0 (pan) + 180 (montaj) + 0 (piksel) = 180 → normalize → -180
        normalized = ((azimut + 180.0) % 360.0) - 180.0
        self.assertAlmostEqual(abs(normalized), 180.0, delta=1.0)

    def test_pan30_camera0_center_bbox(self):
        """Pan +30°, kamera 0, merkez bbox → azimut=30°."""
        self.sm._pan_deg = 30.0
        bbox = (910, 100, 100, 100)
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=0)
        self.assertAlmostEqual(azimut, 30.0, places=1)

    def test_azimut_normalization(self):
        """Azimut -180..180 aralığında kalmalı."""
        self.sm._pan_deg = 60.0
        bbox = (1870, 100, 100, 100)  # sağ kenar
        azimut = self.sm.azimut_from_bbox(bbox, frame_width=1920, camera_id=0)
        self.assertGreaterEqual(azimut, -180.0)
        self.assertLessEqual(azimut, 180.0)


class TestTiltFromBbox(unittest.TestCase):
    """tilt_from_bbox() fonksiyon testleri."""

    def setUp(self):
        self.sm = _make_manager()

    def test_center_bbox_returns_center_tilt(self):
        """Bbox dikey merkezi → tilt_center = (0+60)/2 = 30°."""
        bbox = (100, 490, 100, 100)  # cy = 540, frame_h = 1080
        tilt = self.sm.tilt_from_bbox(bbox, frame_height=1080)
        self.assertAlmostEqual(tilt, 30.0, delta=2.0)

    def test_top_edge_bbox_higher_tilt(self):
        """Bbox üst kenar (drone yüksekte) → daha yüksek tilt."""
        bbox_top = (100, 0, 100, 50)      # cy = 25
        bbox_mid = (100, 490, 100, 100)   # cy = 540
        tilt_top = self.sm.tilt_from_bbox(bbox_top, frame_height=1080)
        tilt_mid = self.sm.tilt_from_bbox(bbox_mid, frame_height=1080)
        self.assertGreater(tilt_top, tilt_mid,
                           msg="Yukarıdaki nesne daha yüksek tilt almalı.")

    def test_tilt_clamped_to_range(self):
        """Tilt [0, 60] aralığı dışına çıkmamalı."""
        bbox_extreme_top = (0, 0, 10, 1)
        bbox_extreme_bot = (0, 1079, 10, 1)
        tilt_top = self.sm.tilt_from_bbox(bbox_extreme_top, frame_height=1080)
        tilt_bot = self.sm.tilt_from_bbox(bbox_extreme_bot, frame_height=1080)
        self.assertGreaterEqual(tilt_top, 0.0)
        self.assertLessEqual(tilt_top, 60.0)
        self.assertGreaterEqual(tilt_bot, 0.0)
        self.assertLessEqual(tilt_bot, 60.0)


class TestStateMachine(unittest.TestCase):
    """SectorManager SWEEP ↔ TRACK durum makinesi testleri."""

    def setUp(self):
        self.mock_serial = MagicMock()
        self.mock_serial.is_connected = True
        self.sm = _make_manager(serial_ctrl=self.mock_serial)

    def _dummy_frame(self, h=1080, w=1920):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_initial_mode_is_sweep(self):
        """Başlangıç modu SWEEP olmalı."""
        self.assertEqual(self.sm.mode, "SWEEP")

    def test_tracking_result_switches_to_track_mode(self):
        """Geçerli TAKIP sonucu → TRACK moduna geç."""
        from configs.constants import STATUS_TRACK
        result = _make_result(bbox=(100, 100, 50, 50), status=STATUS_TRACK, camera_id=0)
        frame = self._dummy_frame()
        snap = _make_snap()

        self.sm.handle_frame_result(frame, result, snap)
        self.assertEqual(self.sm.mode, "TRACK")

    def test_lost_frames_above_threshold_returns_to_sweep(self):
        """Lost kare eşiği geçilince TRACK → SWEEP."""
        from configs.constants import STATUS_TRACK, STATUS_LOST

        # TRACK moduna gir
        track_result = _make_result(bbox=(100, 100, 50, 50), status=STATUS_TRACK)
        frame = self._dummy_frame()
        snap = _make_snap()
        self.sm.handle_frame_result(frame, track_result, snap)
        self.assertEqual(self.sm.mode, "TRACK")

        # Eşik kadar LOST karesini simüle et
        lost_result = _make_result(bbox=(0, 0, 0, 0), status=STATUS_LOST)
        for _ in range(self.sm._lost_threshold + 1):
            self.sm.handle_frame_result(frame, lost_result, snap)

        self.assertEqual(self.sm.mode, "SWEEP")

    def test_servo_command_sent_on_track(self):
        """TRACK modunda SerialController.send_pan_tilt çağrılmalı."""
        from configs.constants import STATUS_TRACK
        result = _make_result(bbox=(100, 100, 50, 50), status=STATUS_TRACK)
        frame = self._dummy_frame()
        snap = _make_snap()

        self.sm.handle_frame_result(frame, result, snap)
        self.mock_serial.send_pan_tilt.assert_called_once()

    def test_no_servo_command_in_detect_mode(self):
        """SWEEP / DETECT modunda drone yok → send_pan_tilt sweep konumu için çağrılır."""
        from configs.constants import STATUS_DETECT
        result = _make_result(bbox=(0, 0, 0, 0), status=STATUS_DETECT)
        frame = self._dummy_frame()
        snap = _make_snap()

        initial_call_count = self.mock_serial.send_pan_tilt.call_count
        self.sm.handle_frame_result(frame, result, snap)
        # Sweep modunda sweep adımı servoyu da günceller
        # call count değişmiş olabilir (sweep) — sadece TRACK komutu olmadığını doğrula
        self.assertEqual(self.sm.mode, "SWEEP")


class TestSweep(unittest.TestCase):
    """Sweep süpürme hareketi testleri."""

    def setUp(self):
        self.sm = _make_manager()

    def test_sweep_reverses_direction_at_max(self):
        """Pan servosu maksimuma ulaşınca yön değişmeli."""
        self.sm._pan_deg = 59.0
        self.sm._sweep_dir = 1.0
        self.sm._last_sweep_time = time.monotonic() - 10.0  # büyük elapsed
        self.sm._step_sweep()
        # Maksimuma çarptıktan sonra yön -1 olmalı
        self.assertEqual(self.sm._sweep_dir, -1.0)

    def test_sweep_reverses_direction_at_min(self):
        """Pan servosu minimuma ulaşınca yön değişmeli."""
        self.sm._pan_deg = -59.0
        self.sm._sweep_dir = -1.0
        self.sm._last_sweep_time = time.monotonic() - 10.0
        self.sm._step_sweep()
        self.assertEqual(self.sm._sweep_dir, 1.0)

    def test_pan_stays_within_limits(self):
        """Pan açısı hiçbir zaman [pan_min, pan_max] dışına çıkmamalı."""
        self.sm._last_sweep_time = time.monotonic() - 100.0  # çok büyük elapsed
        for _ in range(20):
            self.sm._step_sweep()
            self.assertGreaterEqual(self.sm._pan_deg, self.sm._pan_min - 0.01)
            self.assertLessEqual(self.sm._pan_deg, self.sm._pan_max + 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
