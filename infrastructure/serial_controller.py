"""
serial_controller.py — ESP32 ile UART (USB-VCP) iletişimi.

CURSOR İÇİN BAĞLAM:
    SerialController, SectorManager tarafından pan/tilt servo komutları için kullanılır.
    Komut formatı: "PAN:<deg>,TILT:<deg>\\n"   (ASCII, tek satır)
    Thread-safe: tüm yazma işlemleri dahili Lock ile korunur.
    Auto-reconnect: StreamManager'daki reconnect deseniyle aynı yapı.
    main.py bu sınıfı oluşturur ve SectorManager'a enjekte eder.
    settings.yaml → serial bölümü konfigürasyon sağlar.

Kullanım:
    ctrl = SerialController()
    ctrl.connect()
    ctrl.send_pan_tilt(pan_deg=45.0, tilt_deg=20.0)
    ctrl.disconnect()

Donanım Notu (plan.txt § 2.1):
    - CP2102 (USB VID=0x10C4) veya CH340 (USB VID=0x1A86) üzerinden bağlanır.
    - ESP32 UART2 (GPIO16 RX, GPIO17 TX) değil; USB-VCP (UART0) kullanılır.
    - Baud: 115200, 8N1 — ESP32 firmware ile uyumlu olmalı.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from configs import settings
from utils.logger import get_logger

if TYPE_CHECKING:
    import serial as _serial_module

log = get_logger(__name__)

# ESP32 DevKit üzerinde yaygın USB-VCP bridge çiplerinin VID'leri
_KNOWN_ESP32_VIDS: frozenset[int] = frozenset({
    0x10C4,  # Silicon Labs CP2102 / CP2104
    0x1A86,  # QinHeng CH340 / CH341
    0x0403,  # FTDI FT232 (bazı geliştirme kartlarında)
})


def _auto_detect_port() -> str | None:
    """
    Bilinen ESP32 USB-VCP bridge VID'lerine göre seri portu otomatik tespit eder.

    Returns:
        Tespit edilen port adı ("/dev/ttyUSB0", "COM3" vb.) veya None.
    """
    try:
        import serial.tools.list_ports  # type: ignore[import]
        for port_info in serial.tools.list_ports.comports():
            if port_info.vid is not None and port_info.vid in _KNOWN_ESP32_VIDS:
                log.info(
                    "ESP32 portu otomatik tespit: %s  (VID=0x%04X, PID=0x%04X)",
                    port_info.device,
                    port_info.vid or 0,
                    port_info.pid or 0,
                )
                return port_info.device
    except Exception as exc:
        log.debug("Port otomatik tespit başarısız: %s", exc)
    return None


class SerialController:
    """
    ESP32'ye UART üzerinden pan/tilt servo komutları gönderen thread-safe sarmalayıcı.

    Args:
        port:     Seri port adı. None ise settings.yaml, sonra otomatik tespit.
        baudrate: İletişim hızı. None ise settings.yaml değeri (varsayılan 115200).
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int | None = None,
    ) -> None:
        cfg = settings.get("serial", {})

        self._baudrate: int = baudrate or int(cfg.get("baudrate", 115200))
        self._timeout_sec: float = int(cfg.get("timeout_ms", 100)) / 1000.0
        self._reconnect_attempts: int = int(cfg.get("reconnect_attempts", 5))

        # Port öncelik sırası: argüman > settings.yaml > otomatik tespit
        self._port: str | None = port or cfg.get("port") or None

        self._serial: "_serial_module.Serial | None" = None
        self._lock = threading.Lock()
        self._connected = False

    # ── Bağlantı Yönetimi ─────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Seri port bağlantısını açar; gerekirse otomatik port tespiti yapar.

        Returns:
            True: Bağlantı kuruldu.
            False: Tüm denemeler başarısız oldu.
        """
        if self._connected:
            return True

        if self._port is None:
            log.info("SerialController: port belirtilmedi, otomatik tespit deneniyor...")
            self._port = _auto_detect_port()

        if self._port is None:
            log.error(
                "SerialController: ESP32 portu bulunamadı. "
                "settings.yaml → serial.port ayarını kontrol edin."
            )
            return False

        for attempt in range(1, self._reconnect_attempts + 1):
            try:
                import serial  # type: ignore[import]

                self._serial = serial.Serial(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=self._timeout_sec,
                    write_timeout=self._timeout_sec,
                )
                self._connected = True
                log.info(
                    "SerialController bağlandı: %s @ %d baud",
                    self._port,
                    self._baudrate,
                )
                return True

            except Exception as exc:
                log.warning(
                    "Seri port bağlantı denemesi %d/%d başarısız: %s",
                    attempt,
                    self._reconnect_attempts,
                    exc,
                )
                if attempt < self._reconnect_attempts:
                    time.sleep(1.0)

        log.error(
            "SerialController: %d denemede %s portuna bağlanılamadı.",
            self._reconnect_attempts,
            self._port,
        )
        return False

    def disconnect(self) -> None:
        """Seri port bağlantısını güvenli biçimde kapatır."""
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._connected = False
        log.info("SerialController bağlantısı kesildi.")

    @property
    def is_connected(self) -> bool:
        """Bağlantı aktif mi?"""
        return (
            self._connected
            and self._serial is not None
            and self._serial.is_open
        )

    @property
    def port(self) -> str | None:
        """Kullanılan seri port adı."""
        return self._port

    # ── Komut Gönderimi ───────────────────────────────────────────────────────

    def send_pan_tilt(self, pan_deg: float, tilt_deg: float) -> bool:
        """
        ESP32'ye pan/tilt komutu gönderir.

        Komut formatı: "PAN:<pan_deg>,TILT:<tilt_deg>\\n"
        Örnek: "PAN:45.0,TILT:20.0\\n"

        Args:
            pan_deg:  Pan açısı (derece). Pozitif = saat yönü.
            tilt_deg: Tilt açısı (derece). Artış = yukarı.

        Returns:
            True: Komut başarıyla gönderildi.
            False: Bağlantı yok veya yazma hatası.
        """
        if not self.is_connected:
            if not self._try_reconnect():
                return False

        cmd = f"PAN:{pan_deg:.1f},TILT:{tilt_deg:.1f}\n"
        with self._lock:
            try:
                self._serial.write(cmd.encode("ascii"))  # type: ignore[union-attr]
                log.debug("Servo komutu: %s", cmd.strip())
                return True
            except Exception as exc:
                log.warning("Seri yazma hatası: %s — yeniden bağlanılıyor.", exc)
                self._connected = False
                return False

    def send_raw(self, command: str) -> bool:
        """
        Ham ASCII komutu gönderir.
        Test ve debug senaryolarında kullanılır (T1 mock testi).

        Args:
            command: Gönderilecek komut (\\n eklenmemişse otomatik eklenir).

        Returns:
            True: Gönderim başarılı.
            False: Bağlantı yok veya yazma hatası.
        """
        if not self.is_connected:
            if not self._try_reconnect():
                return False

        if not command.endswith("\n"):
            command += "\n"

        with self._lock:
            try:
                self._serial.write(command.encode("ascii"))  # type: ignore[union-attr]
                log.debug("Ham komut gönderildi: %r", command.strip())
                return True
            except Exception as exc:
                log.warning("Seri raw yazma hatası: %s", exc)
                self._connected = False
                return False

    # ── İç Yardımcılar ────────────────────────────────────────────────────────

    def _try_reconnect(self) -> bool:
        """Bağlantı koptuğunda yeniden bağlanmayı dener."""
        log.info("SerialController yeniden bağlanıyor...")
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._connected = False
        return self.connect()

    def __repr__(self) -> str:
        return (
            f"SerialController("
            f"port={self._port!r}, "
            f"baudrate={self._baudrate}, "
            f"connected={self.is_connected}"
            f")"
        )
