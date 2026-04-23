"""
tracker_pool.py — Çoklu hedef (MOT) hibrit takipçi havuzu.

Tek karede bir YOLO geçişi; IoU ile tespit–takipçi eşlemesi; hedef başına
``HybridTracker`` (ITracker + hibrit durum makinesi). Yalnızca inference
thread’den çağrılır.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from configs import settings
from configs.constants import STATUS_LOST
from core.interfaces.idetector import Detection, IDetector
from core.trackers.hybrid_tracker import HybridFrameResult, HybridTracker, TargetData
from utils.logger import get_logger

log = get_logger(__name__)


def _centroid(
    b: tuple[int, int, int, int],
) -> tuple[float, float]:
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


class TrackerPool:
    """
    ``HybridTracker`` sözlüğü: ``dict[str, HybridTracker]``; paylaşılan ``IDetector``
    (YOLO) tek sefer; uzamsal eşleme (IoU, merkez mesafesi yedek).
    """

    def __init__(self, detector: IDetector) -> None:
        self._detector = detector
        self._trackers: dict[str, HybridTracker] = {}
        self._next_id: int = 0
        self._last_boxes: dict[str, tuple[int, int, int, int]] = {}
        self._last_seen: dict[str, tuple[int, int, int, int]] = {}
        self._lost_streak: dict[str, int] = {}

        tcfg = settings.get("tracker", {})
        # mot_match_iou / mot_match_centroid_max_px: geriye uyum (eski config)
        self._hungarian_max_dist_px: float = float(
            tcfg.get("mot_hungarian_max_dist_px", 200.0),
        )
        self._lost_remove: int = int(tcfg.get("mot_lost_remove_frames", 45))
        self._max_tracks: int = int(tcfg.get("mot_max_tracks", 8))

    def _alloc_id(self) -> str:
        self._next_id += 1
        return f"Target-{self._next_id}"

    def _anchor(self, tid: str) -> tuple[int, int, int, int]:
        b = self._last_boxes.get(tid, (0, 0, 0, 0))
        if b == (0, 0, 0, 0) and tid in self._last_seen:
            return self._last_seen[tid]
        return b

    def _match(
        self,
        dets: list[Detection],
    ) -> tuple[dict[str, Detection], list[Detection]]:
        """
        Macar algoritması: izleyici / tespit merkez mesafeleri (euclidean).
        Eşik üstü çiftler maliyet matrisinde maskelenir.
        """
        tids = list(self._trackers.keys())
        if not tids or not dets:
            return {}, list(dets)

        anchors: list[tuple[int, int, int, int]] = []
        valid_tids: list[str] = []
        for tid in tids:
            anc = self._anchor(tid)
            if anc == (0, 0, 0, 0):
                continue
            valid_tids.append(tid)
            anchors.append(anc)
        if not valid_tids:
            return {}, list(dets)

        t_cent = np.array([_centroid(b) for b in anchors], dtype=np.float64)
        d_cent = np.array(
            [_centroid(d.bbox) for d in dets],
            dtype=np.float64,
        )
        cost = cdist(t_cent, d_cent, metric="euclidean")
        high = float(self._hungarian_max_dist_px)
        big = 1e9
        cost[cost > high] = big

        row_ind, col_ind = linear_sum_assignment(cost)
        out: dict[str, Detection] = {}
        used_d: set[int] = set()
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] >= big * 0.5:
                continue
            tid = valid_tids[ri]
            d = dets[ci]
            anc = self._anchor(tid)
            ca, cb = _centroid(anc), _centroid(d.bbox)
            dist = float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))
            if dist > self._hungarian_max_dist_px + 0.1:
                continue
            out[tid] = d
            used_d.add(ci)

        remaining = [d for i, d in enumerate(dets) if i not in used_d]
        return out, remaining

    def _update_memory(self, tid: str, r: HybridFrameResult) -> None:
        self._last_boxes[tid] = r.bbox
        if r.bbox != (0, 0, 0, 0):
            self._last_seen[tid] = r.bbox
        p = r.primary_target
        if p and p.status == STATUS_LOST and p.bbox == (0, 0, 0, 0):
            self._lost_streak[tid] = self._lost_streak.get(tid, 0) + 1
        else:
            self._lost_streak[tid] = 0

    def _gc(self) -> None:
        to_del: list[str] = [
            tid for tid, n in self._lost_streak.items() if n >= self._lost_remove
        ]
        for tid in to_del:
            self._remove(tid)

    def _remove(self, tid: str) -> None:
        ht = self._trackers.pop(tid, None)
        if ht is not None:
            ht.reset()
        self._last_boxes.pop(tid, None)
        self._last_seen.pop(tid, None)
        self._lost_streak.pop(tid, None)
        log.info("Havuz: hedef kaldırıldı: %s", tid)

    def process(
        self,
        frame: np.ndarray,
        camera_id: int = 0,
        latency_ms: float = 0.0,
    ) -> HybridFrameResult:
        dets: list[Detection] = self._detector.detect(frame)
        if not dets and not self._trackers:
            return HybridFrameResult([], camera_id=camera_id)
        by_tid, new_dets = self._match(dets)
        all_targets: list[TargetData] = []

        for tid in list(self._trackers.keys()):
            asg = by_tid.get(tid)
            r = self._trackers[tid].process(
                frame,
                target_id=tid,
                shared_detections=dets,
                assigned=asg,
                camera_id=camera_id,
                latency_ms=latency_ms,
            )
            for t in r.targets:
                all_targets.append(t)
            self._update_memory(tid, r)

        for det in new_dets:
            if len(self._trackers) >= self._max_tracks:
                break
            tid = self._alloc_id()
            self._trackers[tid] = HybridTracker(self._detector)
            r = self._trackers[tid].process(
                frame,
                target_id=tid,
                shared_detections=dets,
                assigned=det,
                camera_id=camera_id,
                latency_ms=latency_ms,
            )
            for t in r.targets:
                all_targets.append(t)
            self._update_memory(tid, r)
            self._lost_streak[tid] = 0

        self._gc()
        return HybridFrameResult(targets=all_targets, camera_id=camera_id)

    def release(self) -> None:
        for tid in list(self._trackers.keys()):
            self._remove(tid)
        self._next_id = 0

    @property
    def state_summary(self) -> str:
        if not self._trackers:
            return "HAVUZ_BOS"
        st = {self._trackers[k].state for k in self._trackers}
        return ",".join(sorted(st))
