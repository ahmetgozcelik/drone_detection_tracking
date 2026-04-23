"""
test_integration_hardware.py — T5-T13 Donanım Entegrasyon Testleri.

Bu testler gerçek donanım gerektirir. Donanım yoksa pytest.skip ile atlanır.

Çalıştırma (tüm testler):
    pytest tests/test_integration_hardware.py -v -s

Seçimli test:
    pytest tests/test_integration_hardware.py::TestT5ServoSweep -v -s
    pytest tests/test_integration_hardware.py::TestT9SerialIntegration -v -s

Ortam Değişkenleri (opsiyonel):
    SERIAL_PORT=/dev/ttyUSB0      → ESP32 port'u (otomatik tespit yerine)
    CAM_INDEX_A=0                 → İlk kamera indeksi
    CAM_INDEX_B=1                 → İkinci kamera indeksi
    INTEGRATION_TIMEOUT=30        → Dayanıklılık testi süresi (saniye)
"""

from __future__ import annotations

import os
import time
import unittest

import numpy as np


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _get_serial_port() -> str | None:
    """Ortam değişkeninden veya otomatik tespitle port bulur."""
    env_port = os.environ.get("SERIAL_PORT")
    if env_port:
        return env_port
    try:
        from infrastructure.serial_controller import _auto_detect_port
        return _auto_detect_port()
    except Exception:
        return None


def _require_hardware(test_case: unittest.TestCase, what: str) -> None:
    """Donanım yoksa testi atlar."""
    test_case.skipTest(
        f"{what} bulunamadı. Gerçek donanım ile çalıştırın. "
        f"İpucu: SERIAL_PORT ortam değişkenini ayarlayın."
    )


# ── T5: ESP32 Servo Sweep Doğrulaması ─────────────────────────────────────────

class TestT5ServoSweep(unittest.TestCase):
    """
    T5: ESP32'ye bağlanıp PING → PONG kontrolü.
    Ardından RESET komutu ile servoları merkeze götür.
    Seri monitörden servo hareketini gözlemle.
    """

    def setUp(self):
        self.port = _get_serial_port()
        if self.port is None:
            self.skipTest("ESP32 portu bulunamadı (T5 atlandı).")

    def test_ping_pong(self):
        """PING → PONG yanıtı alınmalı (ESP32 firmware çalışıyor)."""
        from infrastructure.serial_controller import SerialController
        ctrl = SerialController(port=self.port)
        connected = ctrl.connect()
        self.assertTrue(connected, f"Port {self.port} bağlantısı başarısız.")

        try:
            import serial
            # Direkt raw okuma için lokal serial handle
            ctrl._serial.write(b"PING\n")
            time.sleep(0.3)
            response = ctrl._serial.readline().decode("ascii").strip()
            self.assertEqual(response, "PONG",
                             f"PING → beklenen PONG, alınan: {response!r}")
        finally:
            ctrl.disconnect()

    def test_reset_command(self):
        """RESET → RESET OK yanıtı alınmalı."""
        from infrastructure.serial_controller import SerialController
        ctrl = SerialController(port=self.port)
        ctrl.connect()

        try:
            ctrl._serial.write(b"RESET\n")
            time.sleep(0.5)
            response = ctrl._serial.readline().decode("ascii").strip()
            self.assertIn("RESET", response, f"RESET yanıtı beklendi, alınan: {response!r}")
        finally:
            ctrl.disconnect()


# ── T7: Kamera Açılma Testi ───────────────────────────────────────────────────

class TestT7CameraOpen(unittest.TestCase):
    """
    T7: Her kamerayı tek başına aç, frame al, kapat.
    USB hub ile her kameranın ayrı device index'e düştüğünü doğrula.
    """

    def _open_camera(self, idx: int) -> bool:
        try:
            import cv2
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                return False
            ok, frame = cap.read()
            cap.release()
            return ok and frame is not None
        except Exception:
            return False

    def test_camera_0_opens(self):
        """Kamera 0 açılıp frame okunabilmeli."""
        cam_idx = int(os.environ.get("CAM_INDEX_A", "0"))
        if not self._open_camera(cam_idx):
            self.skipTest(f"Kamera {cam_idx} açılamadı (T7 atlandı).")

    def test_camera_1_opens(self):
        """Kamera 1 açılıp frame okunabilmeli."""
        cam_idx = int(os.environ.get("CAM_INDEX_B", "1"))
        if not self._open_camera(cam_idx):
            self.skipTest(f"Kamera {cam_idx} açılamadı — tek kamera modunda çalışıyorsunuz.")


# ── T8: Çift Kamera Paralel FPS Testi ────────────────────────────────────────

class TestT8DualCameraFPS(unittest.TestCase):
    """
    T8: CompositeStream ile 2 kameradan paralel frame okuma, FPS > 20 kontrolü.
    """

    MEASURE_SECONDS = 5
    MIN_FPS = 20.0

    def test_composite_stream_fps(self):
        """CompositeStream 5 saniyede en az 20 FPS sağlamalı."""
        import cv2
        cam_a = int(os.environ.get("CAM_INDEX_A", "0"))
        cam_b = int(os.environ.get("CAM_INDEX_B", "1"))

        # Kameraların var olup olmadığını hızlıca kontrol et
        cap_test = cv2.VideoCapture(cam_a)
        if not cap_test.isOpened():
            self.skipTest(f"Kamera {cam_a} bulunamadı (T8 atlandı).")
        cap_test.release()

        cap_test = cv2.VideoCapture(cam_b)
        if not cap_test.isOpened():
            self.skipTest(f"Kamera {cam_b} bulunamadı (T8 — tek kamera modu).")
        cap_test.release()

        from infrastructure.streams.usb_stream import UsbStream
        from infrastructure.streams.composite_stream import CompositeStream

        streams = [UsbStream(cam_a), UsbStream(cam_b)]
        comp = CompositeStream(streams)
        self.assertTrue(comp.open(), "CompositeStream açılamadı.")

        frame_count = 0
        t_start = time.monotonic()

        try:
            while time.monotonic() - t_start < self.MEASURE_SECONDS:
                frame = comp.read()
                if frame is not None:
                    frame_count += 1
        finally:
            comp.release()

        elapsed = time.monotonic() - t_start
        measured_fps = frame_count / elapsed if elapsed > 0 else 0.0
        print(f"\n[T8] Toplam kare: {frame_count}, Süre: {elapsed:.1f}s, FPS: {measured_fps:.1f}")

        self.assertGreaterEqual(
            measured_fps, self.MIN_FPS,
            f"FPS {measured_fps:.1f} < hedef {self.MIN_FPS} — "
            f"MJPEG/720p ayarı ve powered hub'ı kontrol edin."
        )


# ── T9: SerialController + ESP32 Entegrasyon ─────────────────────────────────

class TestT9SerialIntegration(unittest.TestCase):
    """
    T9: Python SerialController → ESP32 → servo dönüşü.
    Servo hareketi gözle doğrulanır; test sadece komut gönderim başarısını ölçer.
    """

    def setUp(self):
        self.port = _get_serial_port()
        if self.port is None:
            self.skipTest("ESP32 portu bulunamadı (T9 atlandı).")

    def test_pan_command_sent(self):
        """PAN:45.0,TILT:30.0 komutu başarıyla gönderilmeli."""
        from infrastructure.serial_controller import SerialController
        ctrl = SerialController(port=self.port)
        connected = ctrl.connect()
        self.assertTrue(connected)

        try:
            result = ctrl.send_pan_tilt(45.0, 30.0)
            time.sleep(0.3)
            self.assertTrue(result, "send_pan_tilt False döndü.")
            # ESP32 yanıtını oku (opsiyonel)
            response = ctrl._serial.readline().decode("ascii", errors="replace").strip()
            print(f"\n[T9] ESP32 yanıtı: {response!r}")
        finally:
            # Merkeze dön
            ctrl.send_raw("RESET")
            time.sleep(0.5)
            ctrl.disconnect()

    def test_pan_latency_under_200ms(self):
        """Komut gönderim gecikmesi < 200ms olmalı."""
        from infrastructure.serial_controller import SerialController
        ctrl = SerialController(port=self.port)
        ctrl.connect()

        try:
            t_start = time.monotonic()
            ctrl.send_pan_tilt(0.0, 30.0)
            elapsed_ms = (time.monotonic() - t_start) * 1000
            print(f"\n[T9] Komut gecikme: {elapsed_ms:.1f}ms")
            self.assertLess(elapsed_ms, 200.0,
                            f"Komut gecikmesi {elapsed_ms:.1f}ms > 200ms hedef.")
        finally:
            ctrl.disconnect()


# ── T13: 30 Dakika Dayanıklılık Testi ────────────────────────────────────────

class TestT13Endurance(unittest.TestCase):
    """
    T13: Sistem INTEGRATION_TIMEOUT saniye (varsayılan 30) kesintisiz çalışmalı.
    Gerçek 30 dakika için INTEGRATION_TIMEOUT=1800 ayarlayın.

    Kontrol kriterleri:
    - Servo başarısızlık sayısı: 0
    - USB kamera disconnect sayısı: log'dan izle
    - RAM kullanımı: 1800MB sınırı altında
    """

    TIMEOUT_SEC = int(os.environ.get("INTEGRATION_TIMEOUT", "30"))

    def setUp(self):
        self.port = _get_serial_port()
        if self.port is None:
            self.skipTest("ESP32 portu bulunamadı (T13 atlandı).")

    def test_servo_continuous_operation(self):
        """Servo TIMEOUT_SEC saniye boyunca hata vermeden çalışmalı."""
        from infrastructure.serial_controller import SerialController
        ctrl = SerialController(port=self.port)
        connected = ctrl.connect()
        self.assertTrue(connected)

        failure_count = 0
        command_count = 0
        t_start = time.monotonic()

        try:
            # Sinüs dalgası sweep simülasyonu
            import math
            while time.monotonic() - t_start < self.TIMEOUT_SEC:
                elapsed = time.monotonic() - t_start
                pan = 60.0 * math.sin(2 * math.pi * elapsed / 4.0)  # 4s period
                tilt = 30.0 + 20.0 * math.sin(2 * math.pi * elapsed / 8.0)

                ok = ctrl.send_pan_tilt(pan, tilt)
                command_count += 1
                if not ok:
                    failure_count += 1
                time.sleep(0.1)  # 10 Hz

        finally:
            ctrl.send_raw("RESET")
            ctrl.disconnect()

        print(
            f"\n[T13] Toplam komut: {command_count}, "
            f"Başarısız: {failure_count}, "
            f"Süre: {self.TIMEOUT_SEC}s"
        )
        self.assertEqual(
            failure_count, 0,
            f"{failure_count} servo komut hatası tespit edildi."
        )

    def test_ram_within_limit(self):
        """RAM kullanımı 1800MB altında olmalı (edge device kısıtı)."""
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil kurulu değil.")

        import psutil
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024 * 1024)
        print(f"\n[T13] Mevcut RAM kullanımı: {ram_mb:.0f}MB")
        self.assertLess(ram_mb, 1800,
                        f"RAM {ram_mb:.0f}MB > 1800MB sınırı.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
