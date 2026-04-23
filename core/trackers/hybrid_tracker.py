"""
hybrid_tracker.py — YOLO + OpenCV hibrit durum makinesi (SOT) ve hedef sarmalama (MOT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from configs import settings
from configs.constants import STATUS_DETECT, STATUS_LOST, STATUS_TRACK
from core.interfaces.idetector import Detection, IDetector
from core.interfaces.itracker import ITracker, TrackResult
from core.trackers.base_trackers import create_tracker
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TargetData:
    """Tek hedef: MOT havuzu çıktısı ve hibrit kare sonuç listesi elemanı."""
    target_id: str
    bbox: tuple[int, int, int, int]
    status: str
    confidence: float = 1.0


@dataclass
class HybridFrameResult:
    """
    Kare sonucu: çoklu hedef (MOT) ve alınan kamera.

    ``bbox`` / ``status`` (primary) alanları: SectorManager, metrik, eski SOT
    yolları ile geriye uyum.
    """
    targets: list[TargetData] = field(default_factory=list)
    camera_id: int = 0

    @property
    def primary_target(self) -> TargetData | None:
        if not self.targets:
            return None
        in_track = [
            t for t in self.targets
            if t.status == STATUS_TRACK and t.bbox != (0, 0, 0, 0)
        ]
        if in_track:
            return max(in_track, key=lambda t: t.confidence)
        in_visible = [t for t in self.targets if t.bbox != (0, 0, 0, 0)]
        if in_visible:
            return max(in_visible, key=lambda t: t.confidence)
        return self.targets[0]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        p = self.primary_target
        return p.bbox if p is not None else (0, 0, 0, 0)

    @property
    def status(self) -> str:
        if not self.targets:
            return STATUS_DETECT
        p = self.primary_target
        return p.status if p is not None else STATUS_DETECT

    @classmethod
    def from_single(
        cls,
        bbox: tuple[int, int, int, int],
        status: str,
        camera_id: int = 0,
        confidence: float = 1.0,
        target_id: str = "Target-0",
    ) -> HybridFrameResult:
        t = TargetData(
            target_id=target_id,
            bbox=bbox,
            status=status,
            confidence=confidence,
        )
        return cls([t], camera_id=camera_id)


class HybridTracker:
    """
    Durum makinesi: DETECTING → TRACKING → LOST → DETECTING

    SOT: ``process(frame)`` (kwargs olmadan, yerel YOLO).

    MOT havuzu: ``shared_detections`` + hedefe özel ``assigned`` (başka hedefe
    ait tespit tüketilmez).
    """

    def __init__(self, detector: IDetector) -> None:
        self._detector = detector
        mode = settings["tracker"]["mode"]
        self._tracker: ITracker = create_tracker(mode)
        self._redetect_interval: int = int(settings["tracker"]["redetect_interval"])
        self._state: str = STATUS_DETECT
        self._track_frame_count: int = 0
        self._lost_frame_count: int = 0
        self._pooled: bool = False

    @property
    def state(self) -> str:
        return self._state

    def _transition(self, new_state: str) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        log.info("Durum geçişi: %s → %s", old, new_state)

    @staticmethod
    def _clip_bbox(
        bbox: tuple[int, int, int, int],
        shape: tuple,
    ) -> tuple[int, int, int, int] | None:
        fh, fw = shape[:2]
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w < 4 or h < 4:
            return None
        return (x, y, w, h)

    @staticmethod
    def _best_detection(dets: list[Detection]) -> Detection:
        return max(dets, key=lambda d: d.confidence)

    def _dets(
        self,
        frame: np.ndarray,
        shared: list[Detection] | None,
    ) -> list[Detection]:
        if shared is not None:
            return list(shared)
        return self._detector.detect(frame)

    def _fr(
        self,
        target_id: str,
        bbox: tuple[int, int, int, int],
        status: str,
        conf: float,
        camera_id: int,
    ) -> HybridFrameResult:
        t = TargetData(
            target_id=target_id,
            bbox=bbox,
            status=status,
            confidence=conf,
        )
        return HybridFrameResult([t], camera_id=camera_id)

    def process(
        self,
        frame: np.ndarray,
        *,
        target_id: str = "Target-0",
        shared_detections: list[Detection] | None = None,
        assigned: Detection | None = None,
        camera_id: int = 0,
    ) -> HybridFrameResult:
        self._pooled = shared_detections is not None
        if self._state == STATUS_DETECT:
            return self._step_detecting(
                frame, target_id, shared_detections, assigned, camera_id,
            )
        if self._state == STATUS_TRACK:
            return self._step_tracking(
                frame, target_id, shared_detections, assigned, camera_id,
            )
        return self._step_lost(
            frame, target_id, shared_detections, assigned, camera_id,
        )

    def _step_detecting(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
    ) -> HybridFrameResult:
        dets = self._dets(frame, shared)
        if self._pooled and assigned is None:
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        if not dets and assigned is None:
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        if assigned is not None:
            det = assigned
        else:
            if not dets:
                return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
            det = self._best_detection(dets)
        bbox = self._clip_bbox(det.bbox, frame.shape)
        if bbox is None:
            log.warning("Tespit bbox frame dışında, atlanıyor: %s", det.bbox)
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        self._tracker.reset()
        if not self._tracker.init(frame, bbox):
            log.warning("Takipçi init başarısız (DETECTING). bbox=%s", bbox)
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        self._transition(STATUS_TRACK)
        self._track_frame_count = 0
        self._lost_frame_count = 0
        return self._fr(target_id, bbox, STATUS_TRACK, float(det.confidence), camera_id)

    def _step_tracking(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
    ) -> HybridFrameResult:
        self._track_frame_count += 1
        tr: TrackResult = self._tracker.update(frame)
        conf = float(tr.confidence) if tr.success else 0.0
        periodic = (
            (not self._pooled)
            and self._redetect_interval > 0
            and self._track_frame_count % self._redetect_interval == 0
        )
        if self._pooled:
            need_yolo = not tr.success
        else:
            need_yolo = (not tr.success) or periodic
        if not need_yolo:
            return self._fr(target_id, tr.bbox, STATUS_TRACK, conf, camera_id)
        dets = self._dets(frame, shared)
        if not dets:
            self._tracker.reset()
            self._transition(STATUS_LOST)
            self._lost_frame_count = 0
            return self._fr(target_id, (0, 0, 0, 0), STATUS_LOST, 0.0, camera_id)
        if not tr.success:
            if self._pooled:
                if assigned is not None:
                    det = assigned
                else:
                    self._transition(STATUS_LOST)
                    self._lost_frame_count = 0
                    return self._fr(target_id, (0, 0, 0, 0), STATUS_LOST, 0.0, camera_id)
            else:
                det = self._best_detection(dets)
            self._tracker.reset()
            if self._tracker.init(frame, det.bbox):
                self._track_frame_count = 0
                return self._fr(
                    target_id, det.bbox, STATUS_TRACK, float(det.confidence), camera_id,
                )
            self._transition(STATUS_LOST)
            self._lost_frame_count = 0
            return self._fr(target_id, (0, 0, 0, 0), STATUS_LOST, 0.0, camera_id)
        return self._fr(target_id, tr.bbox, STATUS_TRACK, conf, camera_id)

    def _step_lost(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
    ) -> HybridFrameResult:
        dets = self._dets(frame, shared)
        reacquire: Detection | None
        if self._pooled:
            reacquire = assigned
        else:
            reacquire = self._best_detection(dets) if dets else None
        if reacquire is not None:
            bbox = self._clip_bbox(reacquire.bbox, frame.shape)
            if bbox is not None:
                self._tracker.reset()
                if self._tracker.init(frame, bbox):
                    self._transition(STATUS_TRACK)
                    self._track_frame_count = 0
                    self._lost_frame_count = 0
                    return self._fr(
                        target_id,
                        bbox,
                        STATUS_TRACK,
                        float(reacquire.confidence),
                        camera_id,
                    )
                log.warning("Takipçi init başarısız (LOST). bbox=%s", bbox)
        self._lost_frame_count += 1
        if (
            self._redetect_interval > 0
            and self._lost_frame_count >= self._redetect_interval
        ):
            self._transition(STATUS_DETECT)
            self._lost_frame_count = 0
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        return self._fr(target_id, (0, 0, 0, 0), STATUS_LOST, 0.0, camera_id)

    def reset(self) -> None:
        self._tracker.reset()
        self._state = STATUS_DETECT
        self._track_frame_count = 0
        self._lost_frame_count = 0
        self._pooled = False
