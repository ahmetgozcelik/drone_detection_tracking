"""
configs paketi — Ayarları tek noktadan yükle ve dağıt.

Kullanım (projede her yerde):
    from configs import settings, C
    threshold = settings["model"]["conf_threshold"]
    log_dir   = C.LOG_DIR
"""

import yaml
from configs.constants import SETTINGS_PATH
from configs import constants as C


def _load_settings() -> dict:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


settings: dict = _load_settings()

__all__ = ["settings", "C"]
