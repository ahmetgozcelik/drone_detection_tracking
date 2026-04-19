"""
main.py — Drone Tespit ve Takip Sistemi giriş noktası.

Sorumluluk:
    Dependency Injection merkezi. Tüm bağımlılıklar burada oluşturulur
    ve ilgili sınıflara enjekte edilir. Bu dosya dışında hiçbir sınıf
    başka bir somut sınıfı doğrudan örneklemez — yalnızca arayüzleri bilir.

Başlatma sırası:
    1. Loglama sistemi devreye girer.
    2. Kritik dizinler ve model dosyası doğrulanır.
    3. QApplication oluşturulur.
    4. MainWindow başlatılır (Pipeline henüz çalışmıyor).
    5. Kullanıcı kaynak seçer → Pipeline o anda başlar.
    6. Uygulama kapanırken MainWindow.closeEvent Pipeline'ı durdurur.

Askeri Operasyon Notu:
    Sistem başlatma ve kapanma olayları, hata durumları ve kaynak
    değişiklikleri audit log'una (logs/drone_tracker.log) yazılır.
    Log dosyaları RotatingFileHandler ile tutulur; eski kayıtlar silinmez.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox

# Loglama ilk import — diğer modüllerden önce başlatılmalı
from utils.logger import get_logger

log = get_logger(__name__)


# ── Başlangıç Doğrulamaları ──────────────────────────────────────────────────

def _verify_environment() -> list[str]:
    """
    Sistem başlamadan önce kritik gereksinimleri kontrol et.

    Returns:
        Hata mesajlarının listesi. Boşsa sistem hazır demektir.
    """
    errors: list[str] = []

    # Python sürümü
    if sys.version_info < (3, 9):
        errors.append(
            f"Python 3.9+ gerekli. Mevcut: {sys.version_info.major}.{sys.version_info.minor}"
        )

    # ONNX model dosyası
    try:
        from configs import settings
        from configs.constants import ROOT_DIR
        model_path = ROOT_DIR / settings["model"]["onnx_path"]
        if not model_path.is_file():
            errors.append(
                f"ONNX model bulunamadı: {model_path}\n"
                f"Çözüm: python -c \"from ultralytics import YOLO; "
                f"YOLO('models/yolov8n.pt').export(format='onnx', dynamic=True)\""
            )
    except Exception as exc:
        errors.append(f"Konfigürasyon yüklenemedi: {exc}")

    # Log dizini yazılabilir mi
    try:
        from configs.constants import LOG_DIR
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        probe = LOG_DIR / ".write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        errors.append(f"Log dizinine yazılamıyor: {exc}")

    return errors


def _log_system_info() -> None:
    """Başlangıçta sistem bilgilerini audit log'una yaz."""
    import platform
    import cv2
    import onnxruntime as ort

    from configs import settings

    log.info("=" * 60)
    log.info("DRONE TESPİT VE TAKİP SİSTEMİ — BAŞLATILIYOR")
    log.info("=" * 60)
    log.info("Platform   : %s %s", platform.system(), platform.release())
    log.info("Python     : %s", sys.version.split()[0])
    log.info("OpenCV     : %s", cv2.__version__)
    log.info("ONNX RT    : %s", ort.__version__)
    log.info("ONNX Provs : %s", ort.get_available_providers())
    log.info("Model      : %s", settings["model"]["onnx_path"])
    log.info("Tracker    : %s", settings["tracker"]["mode"].upper())
    log.info("Conf Eşiği : %.2f", settings["model"]["conf_threshold"])
    log.info("IoU Eşiği  : %.2f", settings["model"]["iou_threshold"])
    log.info("=" * 60)


# ── Kritik Hata Yakalayıcı ───────────────────────────────────────────────────

def _handle_uncaught_exception(
    exc_type: type,
    exc_value: BaseException,
    exc_tb: object,
) -> None:
    """
    Yakalanmamış tüm exception'ları audit log'una yaz ve kullanıcıya göster.
    sys.excepthook olarak atanır.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        log.info("Kullanıcı tarafından durduruldu (KeyboardInterrupt).")
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return

    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("YAKALANMAMIŞ HATA:\n%s", tb_str)

    # QApplication aktifse kullanıcıya dialog göster
    app = QApplication.instance()
    if app is not None:
        QMessageBox.critical(
            None,
            "Kritik Sistem Hatası",
            f"Beklenmeyen bir hata oluştu ve sistem kapatılıyor.\n\n"
            f"Hata: {exc_type.__name__}: {exc_value}\n\n"
            f"Detaylar için log dosyasını inceleyiniz:\n"
            f"logs/drone_tracker.log",
        )


# ── Ana Giriş Noktası ────────────────────────────────────────────────────────

def main() -> int:
    """
    Uygulama giriş noktası.

    Returns:
        Çıkış kodu (0: başarı, 1: hata).
    """
    # Yakalanmamış exception hook — loglama başladıktan hemen sonra
    sys.excepthook = _handle_uncaught_exception

    # Ortam doğrulaması — QApplication'dan önce
    errors = _verify_environment()
    if errors:
        # QApplication henüz yok; konsola yaz ve çık
        for err in errors:
            print(f"[HATA] {err}", file=sys.stderr)
        log.critical("Sistem başlatılamadı. Hatalar:\n%s", "\n".join(errors))
        return 1

    # Sistem bilgilerini logla
    _log_system_info()

    # Qt uygulaması
    app = QApplication(sys.argv)
    app.setApplicationName("Drone Takip Sistemi")
    app.setOrganizationName("OSTİM Teknik Üniversitesi")

    # Ana pencere — Pipeline burada başlamaz; kullanıcı kaynak seçince başlar
    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    log.info("Ana pencere gösterildi. Kullanıcı etkileşimi bekleniyor.")

    # Qt event döngüsü — uygulama burada bloklanır
    exit_code = app.exec_()

    log.info("=" * 60)
    log.info("SİSTEM KAPATILDI. Çıkış kodu: %d", exit_code)
    log.info("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
