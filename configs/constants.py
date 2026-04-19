"""
constants.py — Uygulama genelinde değişmez sabitler.

CURSOR İÇİN BAĞLAM:
    Bu değerler asla runtime'da değişmez; settings.yaml'dan farklıdır.
    Sihirli sayı (magic number) kullanma — her zaman buradan import et.
    Yeni bir sabit ekleyeceksen buraya ekle, dosyalara gömme.
"""

from pathlib import Path

# ── Dizin Yapısı ────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent.resolve()
MODEL_DIR       = ROOT_DIR / "models"
CONFIG_DIR      = ROOT_DIR / "configs"
LOG_DIR         = ROOT_DIR / "logs"
DATA_DIR        = ROOT_DIR / "data"
ASSETS_DIR      = ROOT_DIR / "ui" / "assets"

SETTINGS_PATH   = CONFIG_DIR / "settings.yaml"

# ── Model ───────────────────────────────────────────────────────────────────
ONNX_MODEL_FILE = "yolov8n.onnx"
PT_MODEL_FILE   = "yolov8n.pt"

DRONE_CLASS_ID  = 0          # dataset_structure.txt: tek sınıf = drone
DRONE_LABEL     = "drone"
YOLO_INPUT_SIZE = 640        # YOLOv8 giriş boyutu (px); değiştirme

# ── Takip ───────────────────────────────────────────────────────────────────
TRACKER_KCF     = "kcf"
TRACKER_CSRT    = "csrt"

# ── Thread & Kuyruk ─────────────────────────────────────────────────────────
CAPTURE_THREAD_NAME   = "CaptureThread"
INFERENCE_THREAD_NAME = "InferenceThread"
QUEUE_MAXSIZE         = 2       # Düşük tut → frame drop yerine güncel kare işle

# ── Renk Sabitleri (BGR) ────────────────────────────────────────────────────
COLOR_GREEN  = (0, 255, 0)       # Tespit / aktif takip bbox
COLOR_YELLOW = (0, 255, 255)     # Takip devam ediyor (düşük güven)
COLOR_RED    = (0, 0, 255)       # Takip kaybedildi
COLOR_WHITE  = (255, 255, 255)   # Metin

# ── UI Metin Şablonları ──────────────────────────────────────────────────────
FPS_LABEL       = "FPS: {:.1f}"
LATENCY_LABEL   = "Latency: {:.1f} ms"
COORD_LABEL     = "XY: ({}, {})"
STATUS_DETECT   = "TESPIT"
STATUS_TRACK    = "TAKIP"
STATUS_LOST     = "KAYIP"
STATUS_IDLE     = "BEKLENIYOR"
