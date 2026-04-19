"""
hybrid_tracker.py — YOLO + OpenCV takipçi hibrit durum makinesi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs import settings
from configs.constants import STATUS_DETECT, STATUS_LOST, STATUS_TRACK
from core.interfaces.idetector import Detection, IDetector
from core.interfaces.itracker import ITracker, TrackResult
from core.trackers.base_trackers import create_tracker
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class HybridFrameResult:
    """Tek karelik hibrit çıktı: bbox + UI durum metni (constants.STATUS_*)."""
    bbox: tuple[int, int, int, int]
    status: str


class HybridTracker:
    """
    Durum makinesi: DETECTING → TRACKING → LOST → DETECTING

    - DETECTING: Her karede YOLO; tespit varsa takipçi init → TRACKING.
    - TRACKING: Her karede ITracker.update; başarısızlıkta veya her
      redetect_interval karesinde YOLO; YOLO boşsa → LOST.
    - LOST: Her karede YOLO; tespit varsa → TRACKING; redetect_interval boyunca
      bulunamazsa → DETECTING.
    """

    def __init__(self, detector: IDetector) -> None:
        self._detector = detector
        mode = settings["tracker"]["mode"]
        self._tracker: ITracker = create_tracker(mode)
        self._redetect_interval: int = int(settings["tracker"]["redetect_interval"])
        self._state: str = STATUS_DETECT
        self._track_frame_count: int = 0
        self._lost_frame_count: int = 0

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
        """
        bbox'ı frame sınırlarına kırp. Geçerli alan kalmadıysa None döner.
        CSRT/KCF frame dışı koordinatlarla init edilirse başarısız olur.
        """
        fh, fw = shape[:2]
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w < 4 or h < 4:   # Çok küçük bbox takip edilemez
            return None
        return (x, y, w, h)

    @staticmethod
    def _best_detection(dets: list[Detection]) -> Detection:
        return max(dets, key=lambda d: d.confidence)

    def process(self, frame: np.ndarray) -> HybridFrameResult:
        if self._state == STATUS_DETECT:
            return self._step_detecting(frame)
        if self._state == STATUS_TRACK:
            return self._step_tracking(frame)
        return self._step_lost(frame)

    def _step_detecting(self, frame: np.ndarray) -> HybridFrameResult:
        dets = self._detector.detect(frame)
        if not dets:
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_DETECT)

        det = self._best_detection(dets)
        bbox = self._clip_bbox(det.bbox, frame.shape)
        if bbox is None:
            log.warning("Tespit bbox frame dışında, atlanıyor: %s", det.bbox)
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_DETECT)

        self._tracker.reset()
        if not self._tracker.init(frame, bbox):
            log.warning("Takipçi init başarısız (DETECTING). bbox=%s", bbox)
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_DETECT)

        self._transition(STATUS_TRACK)
        self._track_frame_count = 0
        self._lost_frame_count = 0
        return HybridFrameResult(bbox=bbox, status=STATUS_TRACK)

    def _step_tracking(self, frame: np.ndarray) -> HybridFrameResult:
        self._track_frame_count += 1
        tr: TrackResult = self._tracker.update(frame)

        periodic = (
            self._redetect_interval > 0
            and self._track_frame_count % self._redetect_interval == 0
        )
        need_yolo = (not tr.success) or periodic

        if not need_yolo:
            return HybridFrameResult(bbox=tr.bbox, status=STATUS_TRACK)

        dets = self._detector.detect(frame)
        if not dets:
            self._tracker.reset()
            self._transition(STATUS_LOST)
            self._lost_frame_count = 0
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_LOST)

        if not tr.success:
            det = self._best_detection(dets)
            bbox = det.bbox
            self._tracker.reset()
            if self._tracker.init(frame, bbox):
                self._track_frame_count = 0
                return HybridFrameResult(bbox=bbox, status=STATUS_TRACK)
            self._transition(STATUS_LOST)
            self._lost_frame_count = 0
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_LOST)

        return HybridFrameResult(bbox=tr.bbox, status=STATUS_TRACK)

    def _step_lost(self, frame: np.ndarray) -> HybridFrameResult:
        dets = self._detector.detect(frame)
        if dets:
            det = self._best_detection(dets)
            bbox = self._clip_bbox(det.bbox, frame.shape)
            if bbox is None:
                log.warning("LOST: Tespit bbox frame dışında, atlanıyor.")
            else:
                self._tracker.reset()
                if self._tracker.init(frame, bbox):
                    self._transition(STATUS_TRACK)
                    self._track_frame_count = 0
                    self._lost_frame_count = 0
                    return HybridFrameResult(bbox=bbox, status=STATUS_TRACK)
                log.warning("Takipçi init başarısız (LOST). bbox=%s", bbox)

        self._lost_frame_count += 1
        if (
            self._redetect_interval > 0
            and self._lost_frame_count >= self._redetect_interval
        ):
            self._transition(STATUS_DETECT)
            self._lost_frame_count = 0
            return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_DETECT)

        return HybridFrameResult(bbox=(0, 0, 0, 0), status=STATUS_LOST)

    def reset(self) -> None:
        """Pipeline yeniden başlatırken takipçi ve durumu sıfırla."""
        self._tracker.reset()
        self._state = STATUS_DETECT
        self._track_frame_count = 0
        self._lost_frame_count = 0