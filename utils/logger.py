"""
logger.py — Merkezi loglama servisi.

CURSOR İÇİN BAĞLAM:
    Tüm sınıflar (detector, tracker, stream, pipeline) bu modülden logger alır.
    Her sınıfın kendi logger'ı var: get_logger(__name__) ile alınır.
    Log dosyaları askeri denetim izi gereksinimini karşılamak için rotation ile tutulur.
    settings.yaml → logging bölümü bu modülü konfigüre eder.

Kullanım:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Sistem başlatıldı.")
    log.warning("Takip güveni düştü: %.2f", confidence)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from configs import settings
from configs.constants import LOG_DIR

# Modül başına yalnızca bir kez çalışır
_initialized = False


def _setup_root_logger() -> None:
    """Root logger'ı settings.yaml'a göre yapılandır. Yalnızca bir kez çağrılır."""
    global _initialized
    if _initialized:
        return

    cfg = settings.get("logging", {})
    level_str: str = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    log_dir = Path(cfg.get("log_dir", LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "drone_tracker.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Dönen dosya handler'ı (denetim izi için zorunlu)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=cfg.get("max_file_size_mb", 10) * 1024 * 1024,
        backupCount=cfg.get("backup_count", 3),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Konsol çıktısı (geliştirme için; production'da kapatılabilir)
    if cfg.get("enable_console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    İsimli bir logger döndür. Her modül __name__ ile çağırır.

    Args:
        name: Genellikle __name__ (örn. "core.detectors.yolo_onnx")

    Returns:
        Yapılandırılmış logging.Logger nesnesi.
    """
    _setup_root_logger()
    return logging.getLogger(name)
