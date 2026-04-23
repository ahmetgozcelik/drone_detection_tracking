"""
tracker_pool.py — Çoklu hedef (MOT) hibrit takipçi havuzu.

Tek karede bir YOLO geçişi; IoU ile tespit–takipçi eşlemesi; hedef başına
``HybridTracker`` (ITracker + hibrit durum makinesi). Yalnızca inference
thread’den çağrılır.
"""

from __future__ import annotations

import math

from configs import settings
from configs.constants import STATUS_LOST
from core.interfaces.idetector import Detection, IDetector
from core.trackers.hybrid_tracker import HybridFrameResult, HybridTracker, TargetData
from utils.logger import get_logger

log = get_logger(__name__)


def _box_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(a_x2, b_x2), min(a_y2, b_y2)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    u = float(aw * ah + bw * bh) - inter
    return inter / u if u > 0.0 else 0.0


def _centroid(
    b: tuple[int, int, int, int],
) -> tuple[float, float]:
    x, y, w, h = b
    return (x + w / 2.0, y + h / 2.0)


def _centroid_dist(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int],
) -> float:
    ca, cb = _centroid(a), _centroid(b)
    return float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))


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
        self._match_iou: float = float(tcfg.get("mot_match_iou", 0.12))
        self._match_centroid_max_px: float = float(tcfg.get("mot_match_centroid_max_px", 180.0))
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
        tids = list(self._trackers.keys())
        if not tids or not dets:
            return {}, list(dets)
        used_t, used_d = set(), set()
        out: dict[str, Detection] = {}

        pairs: list[tuple[float, int, int, int]] = []  # (score desc, is_iou, tid_idx, det_idx)
        for di, d in enumerate(dets):
            for tii, tid in enumerate(tids):
                anc = self._anchor(tid)
                if anc == (0, 0, 0, 0):
                    continue
                iouv = _box_iou(anc, d.bbox)
                if iouv >= self._match_iou:
                    pairs.append((iouv, 1, tii, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        for score, _is_iou, tii, di in pairs:
            tid = tids[tii]
            if tid in used_t or di in used_d:
                continue
            out[tid] = dets[di]
            used_t.add(tid)
            used_d.add(di)

        cands: list[tuple[float, int, int]] = []
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            for tii, tid in enumerate(tids):
                if tid in out:
                    continue
                anc = self._anchor(tid)
                if anc == (0, 0, 0, 0):
                    continue
                cands.append((_centroid_dist(anc, d.bbox), tii, di))
        cands.sort(key=lambda p: p[0])
        for dist, tii, di in cands:
            tid = tids[tii]
            if tid in out or di in used_d or dist > self._match_centroid_max_px:
                continue
            out[tid] = dets[di]
            used_t.add(tid)
            used_d.add(di)

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

    def process(self, frame: np.ndarray, camera_id: int = 0) -> HybridFrameResult:
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
