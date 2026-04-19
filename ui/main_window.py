"""
main_window.py — MVVM View katmanı; iş mantığı içermez.

Tüm state, Pipeline.on_result callback'inden gelen (frame, HybridFrameResult, MetricsSnapshot)
parametrelerinden türetilir. Cross-thread UI güncellemesi yalnızca pyqtSignal ile yapılır.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from configs import settings
from configs.constants import (
    COLOR_GREEN,
    COLOR_WHITE,
    COLOR_YELLOW,
    COORD_LABEL,
    STATUS_DETECT,
    STATUS_IDLE,
    STATUS_LOST,
    STATUS_TRACK,
)
from core.engine.pipeline import Pipeline
from core.interfaces import IStream
from core.trackers.hybrid_tracker import HybridFrameResult
from utils.logger import get_logger
from utils.metrics import MetricsSnapshot

log = get_logger(__name__)

_STATUS_STYLESHEET: dict[str, str] = {
    STATUS_TRACK:  "color: #00ff00; font-weight: bold;",
    STATUS_DETECT: "color: #ffff00; font-weight: bold;",
    STATUS_LOST:   "color: #ff0000; font-weight: bold;",
    STATUS_IDLE:   "color: #888888; font-weight: bold;",
}

_BBOX_COLOR: dict[str, tuple[int, int, int]] = {
    STATUS_TRACK:  COLOR_GREEN,
    STATUS_DETECT: COLOR_YELLOW,
    STATUS_LOST:   (0, 0, 100),
}

_BBOX_THICKNESS: int = int(settings.get("ui", {}).get("bbox_thickness", 2))


class MainWindow(QMainWindow):
    """Drone Takip Sistemi ana penceresi (MVVM — View)."""

    _sig_result = pyqtSignal(object, object, object)

    def __init__(self) -> None:
        super().__init__()
        ui_cfg = settings.get("ui", {})
        self.setWindowTitle(ui_cfg.get("window_title", "Drone Takip Sistemi"))
        self.setMinimumSize(880, 520)

        self._pipeline: Pipeline | None = None
        self._last_known_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        self._build_ui()
        self._sig_result.connect(self._on_result_received)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        content = QHBoxLayout()
        root_layout.addLayout(content, stretch=1)

        self._lbl_video = QLabel()
        self._lbl_video.setMinimumSize(640, 480)
        self._lbl_video.setAlignment(Qt.AlignCenter)
        self._lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_video.setStyleSheet("background-color: #000;")
        content.addWidget(self._lbl_video, stretch=1)

        panel = QVBoxLayout()
        panel.setSpacing(10)
        content.addLayout(panel)

        self._lbl_status  = self._make_label("DURUM: ---",      fixed_w=220)
        self._lbl_fps     = self._make_label("FPS: ---",         fixed_w=220)
        self._lbl_latency = self._make_label("Latency: --- ms",  fixed_w=220)
        self._lbl_ram     = self._make_label("RAM: --- MB",      fixed_w=220)
        self._lbl_coord   = self._make_label("XY: (---, ---)",   fixed_w=220)

        for lbl in (self._lbl_status, self._lbl_fps, self._lbl_latency,
                    self._lbl_ram, self._lbl_coord):
            panel.addWidget(lbl)
        panel.addStretch()

        bottom = QHBoxLayout()
        root_layout.addLayout(bottom)

        self._btn_file   = QPushButton("📂 Dosya Ac")
        self._btn_camera = QPushButton("📡 Kamera Baglan")
        self._lbl_conn   = QLabel("Baglanti yok")
        self._lbl_conn.setStyleSheet("color: #888;")

        self._btn_file.setFixedHeight(32)
        self._btn_camera.setFixedHeight(32)

        bottom.addWidget(self._btn_file)
        bottom.addWidget(self._btn_camera)
        bottom.addWidget(self._lbl_conn, stretch=1)

        self._btn_file.clicked.connect(self._on_open_file)
        self._btn_camera.clicked.connect(self._on_connect_camera)

    @staticmethod
    def _make_label(text: str, *, fixed_w: int) -> QLabel:
        lbl = QLabel(text)
        lbl.setFixedWidth(fixed_w)
        lbl.setStyleSheet("color: #ccc; font-family: Consolas, monospace; font-size: 13px;")
        return lbl

    def _start_pipeline(self, stream: IStream) -> None:
        self._stop_pipeline()
        self._pipeline = Pipeline(stream=stream, on_result=self._sig_result.emit)
        try:
            self._pipeline.start()
        except Exception as exc:
            log.exception("Pipeline baslatılamadı: %s", exc)
            QMessageBox.critical(self, "Hata", f"Pipeline baslatılamadı:\n{exc}")
            self._pipeline = None
            self._lbl_conn.setText("Baglanti basarısız")
            self._lbl_conn.setStyleSheet("color: #ff0000;")
            return
        self._lbl_conn.setText("Aktif")
        self._lbl_conn.setStyleSheet("color: #00ff00;")
        log.info("Pipeline baslatıldı.")

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                log.exception("Pipeline durdurulurken hata: %s", exc)
            self._pipeline = None
            self._lbl_conn.setText("Baglanti yok")
            self._lbl_conn.setStyleSheet("color: #888;")

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Video Dosyası Sec", "",
            "Video (*.mp4 *.avi *.mkv *.mov);;Tüm Dosyalar (*)",
        )
        if not path:
            return
        log.info("Dosya secildi: %s", path)
        from infrastructure.streams.file_stream import FileStream
        stream: IStream = FileStream(path)
        self._start_pipeline(stream)

    def _on_connect_camera(self) -> None:
        cfg = settings.get("stream", {})
        source = cfg.get("default_source", "usb")
        log.info("Kamera baglantısı istendi. Kaynak: %s", source)

        if source == "usb":
            from infrastructure.streams.usb_stream import UsbStream
            device_idx = cfg.get("usb_device_index", 0)
            stream: IStream = UsbStream(device_idx)
            self._start_pipeline(stream)

        elif source == "rtsp":
            url = cfg.get("rtsp_url", "")
            if not url:
                url, ok = QInputDialog.getText(
                    self, "RTSP URL", "Kamera akış adresini girin:",
                )
                if not ok or not url.strip():
                    return
            # TODO: RtspStream implementasyonu tamamlandığında aktif et
            # from infrastructure.streams.rtsp_stream import RtspStream
            # stream: IStream = RtspStream(url.strip())
            # self._start_pipeline(stream)
            QMessageBox.information(self, "Bilgi", "RtspStream henüz implemente edilmedi.")

        else:
            QMessageBox.critical(self, "Hata", f"Bilinmeyen kaynak tipi: {source}")

    def _on_result_received(
        self, frame: np.ndarray, result: HybridFrameResult, snap: MetricsSnapshot,
    ) -> None:
        self._update_display(frame, result, snap)

    def _update_display(
        self, frame: np.ndarray, result: HybridFrameResult, snap: MetricsSnapshot,
    ) -> None:
        status = result.status
        bbox   = result.bbox
        has_valid_bbox = bbox != (0, 0, 0, 0)

        if has_valid_bbox:
            self._last_known_bbox = bbox

        self._lbl_status.setText(f"DURUM: {status}")
        self._lbl_status.setStyleSheet(
            _STATUS_STYLESHEET.get(status, _STATUS_STYLESHEET[STATUS_IDLE])
            + " font-family: Consolas, monospace; font-size: 13px;"
        )
        self._lbl_fps.setText(snap.fps_label)
        self._lbl_latency.setText(snap.latency_label)
        self._lbl_ram.setText(f"RAM: {snap.ram_used_mb:.0f} MB")

        if has_valid_bbox:
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            self._lbl_coord.setText(COORD_LABEL.format(cx, cy))
            self._lbl_coord.setStyleSheet(
                "color: #ccc; font-family: Consolas, monospace; font-size: 13px;"
            )
        elif status == STATUS_LOST and self._last_known_bbox != (0, 0, 0, 0):
            lx = self._last_known_bbox[0] + self._last_known_bbox[2] // 2
            ly = self._last_known_bbox[1] + self._last_known_bbox[3] // 2
            self._lbl_coord.setText(f"SON: ({lx}, {ly})")
            self._lbl_coord.setStyleSheet(
                "color: #888; font-family: Consolas, monospace; font-size: 13px;"
            )

        rendered = self._draw_overlay(frame, result)
        self._set_pixmap(rendered)

    def _draw_overlay(self, frame: np.ndarray, result: HybridFrameResult) -> np.ndarray:
        vis    = frame.copy()
        status = result.status
        bbox   = result.bbox
        has_valid_bbox = bbox != (0, 0, 0, 0)

        if has_valid_bbox:
            color = _BBOX_COLOR.get(status, COLOR_GREEN)
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, _BBOX_THICKNESS)
            cv2.putText(
                vis, f"{status}",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
            )
        elif status == STATUS_LOST and self._last_known_bbox != (0, 0, 0, 0):
            x, y, w, h = self._last_known_bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 100), _BBOX_THICKNESS)
            cv2.putText(
                vis, "SON KONUM",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA,
            )
        return vis

    def _set_pixmap(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg   = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(qimg).scaled(
            self._lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._lbl_video.setPixmap(scaled)

    def closeEvent(self, event: QCloseEvent) -> None:
        log.info("Pencere kapatma isteği alındı.")
        self._stop_pipeline()
        log.info("Uygulama kapatıldı.")
        event.accept()
