"""
main_viewmodel.py — Ana pencere için sunum durumu (MVVM ViewModel).

Pipeline çıktısı (frame, HybridFrameResult, MetricsSnapshot) alınır, View'a
gönderilecek metin değerleri, rozet stilleri ve çizim parametreleri üretilir.
View yalnızca bu tiplere göre bound widget ve overlay günceller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from configs.constants import (
    STATUS_DETECT,
    STATUS_IDLE,
    STATUS_LOST,
    STATUS_TRACK,
)
from core.trackers.hybrid_tracker import HybridFrameResult
from utils.metrics import MetricsSnapshot

# Askeri HMI: .cursor/rules/ui.mdc (Renk Kodlama Standardı)
CLR_BADGE: dict[str, tuple[str, str]] = {
    STATUS_TRACK:  ("#00aa00", "#ffffff"),
    STATUS_DETECT: ("#aaaa00", "#000000"),
    STATUS_LOST:   ("#aa0000", "#ffffff"),
    STATUS_IDLE:  ("#1e3a5f", "#64748b"),
}


@dataclass(frozen=True, slots=True)
class ViewTrackLayer:
    """Bir hedefe çizilecek kutu (MOT / çoklu overlay)."""
    bbox: tuple[int, int, int, int]
    color_key: str
    label: str


@dataclass(frozen=True, slots=True)
class OverlayDrawSpec:
    """Video üzeri: birincil / son bilinen (SOT/tek hedef döneminde)."""

    show_primary_bbox: bool
    primary_color_key: str
    bbox: tuple[int, int, int, int]
    label_text: str
    show_last_known: bool
    last_known_bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ViewDisplayState:
    """Bir kare sonrası panel ve video alanına uygulanacak durum."""

    status: str
    badge_bg: str
    badge_fg: str
    fps_value: str
    latency_value: str
    ram_value: str
    coord_value: str
    bbox_caption: str
    last_coord_caption: str
    active_camera_id: int
    track_layers: tuple[ViewTrackLayer, ...]  # MOT: tüm hedefler
    overlay: OverlayDrawSpec


class MainViewModel(QObject):
    """
    Tespit/takip sonuçlarından ViewDisplayState üretir.
    Son bilinen bbox (kayıp modunda gri kutu) durumu burada tutulur.
    """

    state_ready = pyqtSignal(object, object)  # ViewDisplayState, frame (ndarray)

    def __init__(self) -> None:
        super().__init__()
        self._last_known_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    @staticmethod
    def _parse_fps_display(snap: MetricsSnapshot) -> str:
        raw = snap.fps_label
        if ":" in raw:
            return raw.split(":", 1)[-1].strip()
        return raw

    @staticmethod
    def _parse_latency_display(snap: MetricsSnapshot) -> str:
        raw = snap.latency_label
        if ":" in raw:
            return raw.split(":", 1)[-1].replace("ms", "").strip()
        return raw

    @staticmethod
    def _badge_for_status(status: str) -> tuple[str, str]:
        if status in CLR_BADGE:
            return CLR_BADGE[status]
        return CLR_BADGE[STATUS_IDLE]

    @pyqtSlot(object, object, object)
    def on_pipeline_result(
        self,
        frame: object,
        result: object,
        snap: object,
    ) -> None:
        f = _as_ndarray(frame)
        r = _as_hybrid_result(result)
        s = _as_metrics(snap)
        st = self._build_view_state(f, r, s)
        self.state_ready.emit(st, f)

    def _build_view_state(
        self,
        frame: np.ndarray,
        result: HybridFrameResult,
        snap: MetricsSnapshot,
    ) -> ViewDisplayState:
        status = result.status
        bbox = result.bbox
        has_valid_bbox = bbox != (0, 0, 0, 0)
        h, w = frame.shape[0], frame.shape[1]

        if has_valid_bbox:
            self._last_known_bbox = bbox

        bg, fg = self._badge_for_status(status)
        camera_id = int(getattr(result, "camera_id", 0))

        fps_v = self._parse_fps_display(snap)
        lat_v = self._parse_latency_display(snap)
        ram_v = f"{snap.ram_used_mb:.0f}"

        if has_valid_bbox:
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            coord = f"{cx},{cy}"
            bcap = (
                f"bbox: x={bbox[0]} y={bbox[1]}\n"
                f"      w={bbox[2]} h={bbox[3]}"
            )
            ltxt = f"son_xy: ({cx}, {cy})"
        elif status == STATUS_LOST and self._last_known_bbox != (0, 0, 0, 0):
            lx = self._last_known_bbox[0] + self._last_known_bbox[2] // 2
            ly = self._last_known_bbox[1] + self._last_known_bbox[3] // 2
            coord = "---"
            bcap = "bbox: ---"
            ltxt = f"son_xy: ({lx}, {ly})"
        else:
            coord = "---"
            bcap = "bbox: ---"
            ltxt = "son_xy: ---"

        layers = self._track_layers_for_result(result)
        ov = self._build_overlay_spec(
            result,
            w,
            h,
            has_valid_bbox,
        )

        return ViewDisplayState(
            status=status,
            badge_bg=bg,
            badge_fg=fg,
            fps_value=fps_v,
            latency_value=lat_v,
            ram_value=ram_v,
            coord_value=coord,
            bbox_caption=bcap,
            last_coord_caption=ltxt,
            active_camera_id=camera_id,
            track_layers=layers,
            overlay=ov,
        )

    @staticmethod
    def _track_layers_for_result(result: HybridFrameResult) -> tuple[ViewTrackLayer, ...]:
        layers: list[ViewTrackLayer] = []
        for t in result.targets:
            if t.bbox == (0, 0, 0, 0) or t.status == STATUS_LOST:
                continue
            if t.status == STATUS_TRACK:
                ckey = "track"
            else:
                ckey = "detect"
            layers.append(
                ViewTrackLayer(
                    t.bbox,
                    ckey,
                    f"{t.target_id} {t.status}",
                )
            )
        return tuple(layers)

    def _build_overlay_spec(
        self,
        result: HybridFrameResult,
        _frame_w: int,
        _frame_h: int,
        has_valid_bbox: bool,
    ) -> OverlayDrawSpec:
        status = result.status
        bbox = result.bbox

        if has_valid_bbox:
            if status == STATUS_TRACK:
                ckey = "track"
            else:
                ckey = "detect"
            return OverlayDrawSpec(
                show_primary_bbox=True,
                primary_color_key=ckey,
                bbox=bbox,
                label_text=status,
                show_last_known=False,
                last_known_bbox=(0, 0, 0, 0),
            )
        if status == STATUS_LOST and self._last_known_bbox != (0, 0, 0, 0):
            return OverlayDrawSpec(
                show_primary_bbox=False,
                primary_color_key="lost_hint",
                bbox=(0, 0, 0, 0),
                label_text="SON KONUM",
                show_last_known=True,
                last_known_bbox=self._last_known_bbox,
            )
        return OverlayDrawSpec(
            show_primary_bbox=False,
            primary_color_key="track",
            bbox=(0, 0, 0, 0),
            label_text="",
            show_last_known=False,
            last_known_bbox=(0, 0, 0, 0),
        )


def _as_ndarray(frame: object) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame, np.ndarray (veya TaggedFrame) olmalı")
    return frame  # type: ignore[return-value]


def _as_hybrid_result(r: object) -> HybridFrameResult:
    if not isinstance(r, HybridFrameResult):
        raise TypeError("result, HybridFrameResult olmalı")
    return r


def _as_metrics(s: object) -> MetricsSnapshot:
    if not isinstance(s, MetricsSnapshot):
        raise TypeError("MetricsSnapshot beklendi")
    return s
