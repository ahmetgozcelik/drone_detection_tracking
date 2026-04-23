"""
test_serial_controller.py — SerialController birim ve mock testleri (T1).

Donanım gerektirmez. pyserial'ın "loop://" URL'i ile loopback test.

Çalıştırma:
    pytest tests/test_serial_controller.py -v

Gereksinim:
    pip install pyserial

Kapsam:
    - Komut formatı doğrulama: "PAN:<deg>,TILT:<deg>\\n"
    - send_pan_tilt → doğru byte'ların yazıldığı kontrolü (mock serial)
    - send_raw → satır sonu eklenmesi
    - is_connected: bağlantı yokken False
    - auto-reconnect: bağlantı kopunca yeniden deneme
    - _auto_detect_port: VID tarama mantığı (mock list_ports)
"""

from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


_MOCK_SETTINGS = {
    "serial": {
        "port": "loop://",
        "baudrate": 115200,
        "timeout_ms": 100,
        "reconnect_attempts": 3,
        "enabled": True,
    }
}


def _make_controller(**kwargs):
    """Test için SerialController oluşturan yardımcı."""
    with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
        from infrastructure.serial_controller import SerialController
        return SerialController(**kwargs)


class TestSerialControllerCommandFormat(unittest.TestCase):
    """Komut formatı ve byte içeriği testleri."""

    def setUp(self):
        self.mock_serial = MagicMock()
        self.mock_serial.is_open = True
        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            self.ctrl = SerialController(port="loop://", baudrate=115200)
        self.ctrl._serial = self.mock_serial
        self.ctrl._connected = True

    def test_pan_tilt_command_format(self):
        """send_pan_tilt(45.0, 20.0) → 'PAN:45.0,TILT:20.0\\n' yazılmalı."""
        self.ctrl.send_pan_tilt(45.0, 20.0)
        written = self.mock_serial.write.call_args[0][0]
        self.assertEqual(written, b"PAN:45.0,TILT:20.0\n")

    def test_negative_pan(self):
        """Negatif pan açısı doğru formatlanmalı."""
        self.ctrl.send_pan_tilt(-30.5, 15.0)
        written = self.mock_serial.write.call_args[0][0]
        self.assertEqual(written, b"PAN:-30.5,TILT:15.0\n")

    def test_zero_values(self):
        """Sıfır değerler: PAN:0.0,TILT:0.0\\n."""
        self.ctrl.send_pan_tilt(0.0, 0.0)
        written = self.mock_serial.write.call_args[0][0]
        self.assertEqual(written, b"PAN:0.0,TILT:0.0\n")

    def test_send_raw_adds_newline(self):
        """send_raw komutuna otomatik \\n eklenmeli."""
        self.ctrl.send_raw("PING")
        written = self.mock_serial.write.call_args[0][0]
        self.assertEqual(written, b"PING\n")

    def test_send_raw_no_duplicate_newline(self):
        """send_raw'a zaten \\n varsa ek eklenmemeli."""
        self.ctrl.send_raw("RESET\n")
        written = self.mock_serial.write.call_args[0][0]
        self.assertEqual(written, b"RESET\n")

    def test_returns_true_on_success(self):
        """Başarılı yazımda True döndürmeli."""
        result = self.ctrl.send_pan_tilt(10.0, 10.0)
        self.assertTrue(result)


class TestSerialControllerConnection(unittest.TestCase):
    """Bağlantı yönetimi testleri."""

    def test_is_connected_false_before_connect(self):
        """Bağlantı kurulmadan is_connected False olmalı."""
        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            ctrl = SerialController(port="/dev/nonexistent")
        self.assertFalse(ctrl.is_connected)

    def test_disconnect_sets_connected_false(self):
        """disconnect() sonrası is_connected False olmalı."""
        mock_serial = MagicMock()
        mock_serial.is_open = True
        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            ctrl = SerialController(port="loop://")
        ctrl._serial = mock_serial
        ctrl._connected = True
        ctrl.disconnect()
        self.assertFalse(ctrl.is_connected)

    def test_send_pan_tilt_returns_false_when_disconnected(self):
        """Bağlantı yoksa send_pan_tilt False döndürmeli (reconnect da başarısız)."""
        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            ctrl = SerialController(port="/dev/nonexistent_port_123")
        ctrl._connected = False
        ctrl._serial = None
        # Reconnect denemelerini hızlı geçmek için sleep'i mock'la
        with patch("time.sleep"):
            result = ctrl.send_pan_tilt(10.0, 5.0)
        self.assertFalse(result)

    def test_repr_format(self):
        """__repr__ beklenen formatı içermeli."""
        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            ctrl = SerialController(port="COM3", baudrate=115200)
        r = repr(ctrl)
        self.assertIn("COM3", r)
        self.assertIn("115200", r)
        self.assertIn("connected=False", r)


class TestAutoDetectPort(unittest.TestCase):
    """Port otomatik tespit testleri."""

    def test_detects_cp2102_vid(self):
        """VID=0x10C4 (CP2102) → port döndürmeli."""
        mock_port = MagicMock()
        mock_port.vid = 0x10C4
        mock_port.pid = 0xEA60
        mock_port.device = "/dev/ttyUSB0"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            from infrastructure.serial_controller import _auto_detect_port
            detected = _auto_detect_port()
        self.assertEqual(detected, "/dev/ttyUSB0")

    def test_detects_ch340_vid(self):
        """VID=0x1A86 (CH340) → port döndürmeli."""
        mock_port = MagicMock()
        mock_port.vid = 0x1A86
        mock_port.pid = 0x7523
        mock_port.device = "/dev/ttyUSB1"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            from infrastructure.serial_controller import _auto_detect_port
            detected = _auto_detect_port()
        self.assertEqual(detected, "/dev/ttyUSB1")

    def test_returns_none_for_unknown_vid(self):
        """Bilinmeyen VID → None döndürmeli."""
        mock_port = MagicMock()
        mock_port.vid = 0xDEAD
        mock_port.device = "/dev/ttyUSB99"

        with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
            from infrastructure.serial_controller import _auto_detect_port
            detected = _auto_detect_port()
        self.assertIsNone(detected)

    def test_returns_none_when_no_ports(self):
        """Port listesi boşsa None döndürmeli."""
        with patch("serial.tools.list_ports.comports", return_value=[]):
            from infrastructure.serial_controller import _auto_detect_port
            detected = _auto_detect_port()
        self.assertIsNone(detected)


class TestLoopbackIntegration(unittest.TestCase):
    """
    pyserial loop:// URL ile gerçek loopback entegrasyon testi (T1).
    Gönderilen komutun byte seviyesinde doğru olduğunu kontrol eder.
    Seri port driver gerekmez.
    """

    def test_loopback_pan_tilt_format(self):
        """loop:// üzerinden gerçek write → read ile komut doğrulama."""
        try:
            import serial
        except ImportError:
            self.skipTest("pyserial kurulu değil.")

        try:
            ser = serial.serial_for_url("loop://", timeout=0.5)
        except Exception:
            self.skipTest("pyserial loop:// desteklenmiyor.")

        with patch("infrastructure.serial_controller.settings", _MOCK_SETTINGS):
            from infrastructure.serial_controller import SerialController
            ctrl = SerialController(port="loop://")

        ctrl._serial = ser
        ctrl._connected = True

        ctrl.send_pan_tilt(33.5, 12.0)
        received = ser.readline()
        self.assertEqual(received, b"PAN:33.5,TILT:12.0\n")

        ser.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
