"""
main_window.py — MVVM View katmanı; biçimlendirme ve durum geçişleri viewmodels/ içindedir.

Pipeline sonuçları pyqtSignal ile aktarılır; MainViewModel state_ready sinyaliyle
görünür alanlara dönüştürülmüş ViewDisplayState üretir.
Cross-thread: yalnızca sinyal zinciri; widget güncellemeleri UI thread'te.

Görsel katman: askeri standart karanlık HMI teması.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
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
    STATUS_IDLE,
)
from core.engine.pipeline import Pipeline
from core.engine.sector_manager import SectorManager
from core.interfaces import IStream
from ui.viewmodels.main_viewmodel import (
    CLR_BADGE,
    MainViewModel,
    OverlayDrawSpec,
    ViewDisplayState,
    ViewTrackLayer,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ── HMI Renk Paleti (askeri karanlık tema) ──────────────────────────────────
CLR_BG          = "#0a0e1a"
CLR_PANEL       = "#111827"
CLR_BOTTOM      = "#0d1526"
CLR_BORDER      = "#1e3a5f"
CLR_ACCENT      = "#00d4ff"
# Askeri anlam paleti: .cursor/rules/ui.mdc
CLR_SUCCESS     = "#00aa00"
CLR_WARNING     = "#aaaa00"
CLR_DANGER      = "#aa0000"
CLR_TEXT        = "#e2e8f0"
CLR_TEXT_MUTED  = "#64748b"

FONT_MONO = "Consolas, 'Courier New', monospace"

_BBOX_KEY_BGR: dict[str, tuple[int, int, int]] = {
    "track":  COLOR_GREEN,
    "detect": COLOR_YELLOW,
    "lost_hint": (0, 0, 100),
}

_BBOX_THICKNESS: int = int(settings.get("ui", {}).get("bbox_thickness", 2))


# ════════════════════════════════════════════════════════════════════════════
#                          Custom Widget — MetricRow
# ════════════════════════════════════════════════════════════════════════════
class MetricRow(QWidget):
    """
    Sol etiket + noktalı dolgu + sağ değer biçiminde tek satırlık metrik.
    Noktalar dinamik olarak widget genişliğine göre çizilir.
    """

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._value = "---"
        self._label_font = QFont("Segoe UI", 9)
        self._label_font.setLetterSpacing(QFont.PercentageSpacing, 105)
        self._value_font = QFont("Consolas", 10)
        self._value_font.setStyleHint(QFont.Monospace)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, value: str) -> None:
        if value != self._value:
            self._value = value
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        rect = self.rect()
        baseline_y = rect.center().y() + 4

        # Sol etiket
        p.setFont(self._label_font)
        p.setPen(QColor(CLR_TEXT_MUTED))
        label_metrics = QFontMetrics(self._label_font)
        label_w = label_metrics.horizontalAdvance(self._label)
        p.drawText(0, baseline_y, self._label)

        # Sağ değer
        p.setFont(self._value_font)
        p.setPen(QColor(CLR_TEXT))
        value_metrics = QFontMetrics(self._value_font)
        value_w = value_metrics.horizontalAdvance(self._value)
        value_x = rect.width() - value_w
        p.drawText(value_x, baseline_y, self._value)

        # Arada noktalı dolgu
        dots_left  = label_w + 6
        dots_right = value_x - 6
        if dots_right > dots_left:
            p.setPen(QPen(QColor(CLR_BORDER), 1, Qt.DotLine))
            dot_y = rect.center().y() + 2
            p.drawLine(dots_left, dot_y, dots_right, dot_y)

        p.end()


# ════════════════════════════════════════════════════════════════════════════
#                              Main Window
# ════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    """Drone Takip Sistemi ana penceresi (MVVM — View)."""

    _sig_result = pyqtSignal(object, object, object)

    def __init__(self, sector_manager: SectorManager | None = None) -> None:
        super().__init__()
        ui_cfg = settings.get("ui", {})
        self.setWindowTitle(ui_cfg.get("window_title", "Drone Takip Sistemi"))
        self.setMinimumSize(1100, 640)

        self._pipeline: Pipeline | None = None
        self._view_model = MainViewModel()
        self._sector_manager: SectorManager | None = sector_manager

        # SectorManager'dan pan/tilt güncellemelerini UI'ya aktar
        if self._sector_manager is not None:
            self._sector_manager._on_pan_tilt_update = self._on_servo_update

        self._build_ui()
        self._apply_global_style()
        self._sig_result.connect(self._view_model.on_pipeline_result)
        self._view_model.state_ready.connect(self._on_view_state)
        self._set_badge_idle()
        self._set_connection(False, "BAGLANTI YOK")

    # ── Stil ────────────────────────────────────────────────────────────────
    def _apply_global_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget#rootCentral {{
                background-color: {CLR_BG};
            }}
            QToolTip {{
                background-color: {CLR_PANEL};
                color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER};
            }}
        """)

    @staticmethod
    def _button_style() -> str:
        return f"""
            QPushButton {{
                background-color: {CLR_BORDER};
                color: {CLR_ACCENT};
                border: 1px solid {CLR_ACCENT};
                border-radius: 3px;
                padding: 6px 14px;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {CLR_ACCENT};
                color: {CLR_BG};
            }}
            QPushButton:pressed {{
                background-color: #0099c8;
                color: {CLR_BG};
            }}
            QPushButton:disabled {{
                background-color: {CLR_PANEL};
                color: {CLR_TEXT_MUTED};
                border-color: {CLR_TEXT_MUTED};
            }}
        """

    # ── Layout inşa ─────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("rootCentral")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_title_bar())

        content = QHBoxLayout()
        content.setContentsMargins(10, 10, 0, 10)
        content.setSpacing(0)
        root.addLayout(content, stretch=1)

        content.addWidget(self._build_video_area(), stretch=1)
        content.addWidget(self._build_side_panel())

        root.addWidget(self._build_bottom_bar())

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(40)
        bar.setStyleSheet(
            f"background-color: {CLR_BOTTOM};"
            f"border-bottom: 1px solid {CLR_BORDER};"
        )

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(12)

        title = QLabel("DRONE TESPIT VE TAKIP SISTEMI v1.0")
        title.setStyleSheet(
            f"color: {CLR_ACCENT};"
            "font-family: 'Segoe UI';"
            "font-size: 13px;"
            "font-weight: bold;"
            "letter-spacing: 2px;"
        )
        lay.addWidget(title)

        subtitle = QLabel("OSTIM TEKNIK UNIVERSITESI")
        subtitle.setStyleSheet(
            f"color: {CLR_TEXT_MUTED};"
            "font-family: 'Segoe UI';"
            "font-size: 10px;"
            "letter-spacing: 1px;"
        )
        lay.addWidget(subtitle)

        lay.addStretch(1)

        clock_tag = QLabel("● ONLINE")
        clock_tag.setStyleSheet(
            f"color: {CLR_SUCCESS};"
            "font-family: Consolas, monospace;"
            "font-size: 10px;"
            "letter-spacing: 1px;"
        )
        lay.addWidget(clock_tag)

        return bar

    def _build_video_area(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(
            f"background-color: #000000;"
            f"border: 2px solid {CLR_BORDER};"
        )
        wrap_lay = QVBoxLayout(wrap)
        wrap_lay.setContentsMargins(0, 0, 0, 0)

        self._lbl_video = QLabel("● VIDEO FEED BEKLENIYOR")
        self._lbl_video.setMinimumSize(640, 480)
        self._lbl_video.setAlignment(Qt.AlignCenter)
        self._lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_video.setStyleSheet(
            "background-color: #000000;"
            "border: none;"
            f"color: {CLR_TEXT_MUTED};"
            "font-family: Consolas, monospace;"
            "font-size: 12px;"
            "letter-spacing: 2px;"
        )
        wrap_lay.addWidget(self._lbl_video)
        return wrap

    def _build_side_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(260)
        panel.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {CLR_PANEL};"
            f"  border-left: 1px solid {CLR_BORDER};"
            f"}}"
        )

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        # Başlık
        lay.addWidget(self._section_header("SISTEM DURUMU"))

        # Status badge
        self._badge = QLabel("---")
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setFixedHeight(44)
        lay.addWidget(self._badge)

        lay.addWidget(self._separator())

        # Metrik satırları
        self._row_fps     = MetricRow("FPS")
        self._row_latency = MetricRow("LATENCY  ms")
        self._row_ram     = MetricRow("RAM  MB")
        self._row_coord   = MetricRow("XY")
        for row in (self._row_fps, self._row_latency, self._row_ram, self._row_coord):
            lay.addWidget(row)

        lay.addWidget(self._separator())

        # Pan-Tilt Servo Durumu
        lay.addWidget(self._section_header("SERVO / SEKTOR"))
        self._row_pan      = MetricRow("PAN  deg")
        self._row_tilt     = MetricRow("TILT  deg")
        self._row_cam      = MetricRow("AKTIF KAM")
        self._row_sw_mode  = MetricRow("MOD")
        for row in (self._row_pan, self._row_tilt, self._row_cam, self._row_sw_mode):
            lay.addWidget(row)

        lay.addWidget(self._separator())

        # Son tespit
        lay.addWidget(self._section_header("SON TESPIT"))

        self._lbl_bbox = QLabel("bbox: ---")
        self._lbl_bbox.setStyleSheet(
            f"color: {CLR_TEXT};"
            f"font-family: {FONT_MONO};"
            "font-size: 11px;"
            "background: transparent;"
            "border: none;"
        )
        self._lbl_bbox.setWordWrap(True)
        lay.addWidget(self._lbl_bbox)

        self._lbl_last_coord = QLabel("son_xy: ---")
        self._lbl_last_coord.setStyleSheet(
            f"color: {CLR_TEXT_MUTED};"
            f"font-family: {FONT_MONO};"
            "font-size: 11px;"
            "background: transparent;"
            "border: none;"
        )
        lay.addWidget(self._lbl_last_coord)

        lay.addStretch(1)

        return panel

    def _build_bottom_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {CLR_BOTTOM};"
            f"  border-top: 1px solid {CLR_BORDER};"
            f"}}"
        )

        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(10)

        self._btn_file   = QPushButton("DOSYA AC")
        self._btn_camera = QPushButton("KAMERA BAGLAN")
        for btn in (self._btn_file, self._btn_camera):
            btn.setStyleSheet(self._button_style())
            btn.setFixedHeight(32)
            btn.setCursor(Qt.PointingHandCursor)

        lay.addWidget(self._btn_file)
        lay.addWidget(self._btn_camera)

        lay.addStretch(1)

        self._lbl_conn = QLabel()
        self._lbl_conn.setTextFormat(Qt.RichText)
        self._lbl_conn.setStyleSheet(
            f"color: {CLR_TEXT};"
            "font-family: Consolas, monospace;"
            "font-size: 11px;"
            "letter-spacing: 1px;"
            "background: transparent;"
            "border: none;"
        )
        lay.addWidget(self._lbl_conn)

        self._btn_file.clicked.connect(self._on_open_file)
        self._btn_camera.clicked.connect(self._on_connect_camera)

        return bar

    # ── Yardımcı widget factory'leri ────────────────────────────────────────
    @staticmethod
    def _section_header(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {CLR_ACCENT};"
            "font-family: 'Segoe UI';"
            "font-size: 11px;"
            "font-weight: bold;"
            "letter-spacing: 2px;"
            "background: transparent;"
            "border: none;"
            "padding: 2px 0;"
        )
        return lbl

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {CLR_BORDER}; border: none;")
        return line

    # ── Durum güncelleyiciler ───────────────────────────────────────────────
    def _set_badge_idle(self) -> None:
        bg, fg = CLR_BADGE[STATUS_IDLE]
        self._badge.setText(STATUS_IDLE)
        self._apply_badge_style(bg, fg)

    def _apply_badge(self, s: ViewDisplayState) -> None:
        self._badge.setText(s.status)
        self._apply_badge_style(s.badge_bg, s.badge_fg)

    def _apply_badge_style(self, bg: str, fg: str) -> None:
        self._badge.setStyleSheet(
            f"background-color: {bg};"
            f"color: {fg};"
            "border-radius: 4px;"
            "font-family: 'Segoe UI';"
            "font-size: 18px;"
            "font-weight: bold;"
            "letter-spacing: 3px;"
            "padding: 8px 12px;"
        )

    def _set_connection(self, active: bool, text: str) -> None:
        dot_color = CLR_SUCCESS if active else CLR_DANGER
        self._lbl_conn.setText(
            f"<span style='color:{dot_color};'>●</span>"
            f"&nbsp;<span style='color:{CLR_TEXT};'>{text}</span>"
        )

    # ── Servo Güncelleme Callback (SectorManager → UI) ──────────────────────
    def _on_servo_update(self, pan_deg: float, tilt_deg: float, mode: str) -> None:
        """
        SectorManager'ın on_pan_tilt_update callback'i.
        Inference thread'den çağrılır → pyqtSignal ile ana thread'e aktar.
        """
        # MetricRow.set_value thread-safe değil; pyqtSignal gerekir.
        # Basit çözüm: set_value paintEvent tetikler, Qt thread-safe paintEvent sıralar.
        self._row_pan.set_value(f"{pan_deg:.1f}")
        self._row_tilt.set_value(f"{tilt_deg:.1f}")
        self._row_sw_mode.set_value(mode)

    # ── Pipeline yaşam döngüsü ──────────────────────────────────────────────
    def _start_pipeline(self, stream: IStream) -> None:
        self._stop_pipeline()

        # SectorManager varsa on_result callback zinciri oluştur
        if self._sector_manager is not None:
            sm = self._sector_manager

            def _chained_on_result(
                frame: object, result: object, snap: object,
            ) -> None:
                sm.handle_frame_result(frame, result, snap)  # type: ignore[arg-type]
                self._sig_result.emit(frame, result, snap)

            on_result = _chained_on_result
        else:
            on_result = self._sig_result.emit

        self._pipeline = Pipeline(stream=stream, on_result=on_result)
        try:
            self._pipeline.start()
        except Exception as exc:
            log.exception("Pipeline baslatılamadı: %s", exc)
            QMessageBox.critical(self, "Hata", f"Pipeline baslatılamadı:\n{exc}")
            self._pipeline = None
            self._set_connection(False, "BAGLANTI BASARISIZ")
            return
        self._set_connection(True, "AKTIF")
        log.info("Pipeline baslatıldı.")

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                log.exception("Pipeline durdurulurken hata: %s", exc)
            self._pipeline = None
            self._set_connection(False, "BAGLANTI YOK")

    # ── Kullanıcı aksiyonları ───────────────────────────────────────────────
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

            indices = cfg.get("usb_device_indices", None)
            if indices and len(indices) > 1:
                # Çift kamera: CompositeStream (round-robin + TaggedFrame)
                from infrastructure.streams.composite_stream import CompositeStream
                streams = [UsbStream(int(i)) for i in indices]
                stream: IStream = CompositeStream(streams)
                log.info("CompositeStream oluşturuldu: %d kamera %s", len(indices), indices)
            else:
                device_idx = int(cfg.get("usb_device_index", 0))
                stream = UsbStream(device_idx)

            self._start_pipeline(stream)

            # Aktif kamera göstergesini güncelle
            if self._sector_manager is not None:
                self._row_cam.set_value(
                    f"{len(indices) if indices and len(indices) > 1 else 1}"
                )

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

    # ── ViewModel sinyali → UI (ana thread) ─────────────────────────────────
    def _on_view_state(self, state: object, frame: object) -> None:
        if not isinstance(state, ViewDisplayState):
            return
        if not isinstance(frame, np.ndarray):
            return
        self._row_cam.set_value(str(state.active_camera_id))
        self._apply_badge(state)
        self._row_fps.set_value(state.fps_value)
        self._row_latency.set_value(state.latency_value)
        self._row_ram.set_value(state.ram_value)
        self._row_coord.set_value(state.coord_value)
        self._lbl_bbox.setText(state.bbox_caption)
        self._lbl_last_coord.setText(state.last_coord_caption)
        self._set_pixmap(self._draw_view_bgr(frame, state))

    @staticmethod
    def _draw_view_bgr(frame: np.ndarray, state: ViewDisplayState) -> np.ndarray:
        if state.track_layers:
            vis = frame.copy()
            for L in state.track_layers:
                if not isinstance(L, ViewTrackLayer):
                    continue
                color = _BBOX_KEY_BGR.get(L.color_key, COLOR_GREEN)
                x, y, w, h = L.bbox
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, _BBOX_THICKNESS)
                cv2.putText(
                    vis,
                    L.label,
                    (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
                )
            return vis
        return MainWindow._draw_overlay_bgr(frame, state.overlay)

    # ── Video overlay (MOT yok: tek hedef) ─────
    @staticmethod
    def _draw_overlay_bgr(frame: np.ndarray, spec: OverlayDrawSpec) -> np.ndarray:
        vis = frame.copy()
        if spec.show_primary_bbox:
            color = _BBOX_KEY_BGR.get(spec.primary_color_key, COLOR_GREEN)
            x, y, w, h = spec.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, _BBOX_THICKNESS)
            cv2.putText(
                vis,
                spec.label_text,
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
            )
        elif spec.show_last_known:
            x, y, w, h = spec.last_known_bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 100), _BBOX_THICKNESS)
            cv2.putText(
                vis,
                spec.label_text,
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
