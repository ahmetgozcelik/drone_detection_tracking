"""
hybrid_tracker.py — YOLO-güdümlü hibrit asenkron takip + KCF kurtarma (5 kare) ve
Kalman (8B) pürüzsüzleştirme. MOT: ``TrackerPool`` hedef başına ``HybridTracker``.

Birincil: her kare YOLO eşleşmesi. YOLO yok: KCF ile en fazla N ardışık kare;
aşılırsa KAYIP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from configs import settings
from configs.constants import STATUS_DETECT, STATUS_LOST, STATUS_TRACK
from core.interfaces.idetector import Detection, IDetector
from core.interfaces.itracker import TrackResult
from core.trackers.base_trackers import KcfTracker
from core.trackers.kalman_predictor import KalmanPredictor
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

    ``bbox`` / ``status`` (primary) alanları: SectorManager, metrik, eski yollar
    ile geriye uyum.
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
    YOLO-güdümlü KCF: YOLO eşleşmesi varken Kalman düzeltir + KCF her eşleşmede
    yeniden yüklenir. Eşleşme yokken (en fazla N kare) KCF ``update`` ile
    kurtarma; aşımında KAYIP.

    SOT sınıfı (``CsrtTracker``) bu mimaride birincil kare hızı için
    çağrılmaz; sadece ``KcfTracker`` yedek yol olarak kullanılır.
    """

    def __init__(self, detector: IDetector) -> None:
        self._detector = detector
        tcfg = settings["tracker"]
        s_cfg = settings.get("stream", {})
        self._target_fps: float = float(s_cfg.get("target_fps", 30.0))
        self._redetect_interval: int = int(tcfg.get("redetect_interval", 60))
        self._max_fallback_frames: int = int(
            tcfg.get("kcf_fallback_max_frames", 5),
        )
        self._kcf: KcfTracker = KcfTracker()
        self._state: str = STATUS_DETECT
        self._track_frame_count: int = 0
        self._lost_frame_count: int = 0
        self._pooled: bool = False
        self._kalman: KalmanPredictor | None = None
        self._fallback_count: int = 0

    @property
    def state(self) -> str:
        return self._state

    def _transition(self, new_state: str) -> None:
        if new_state == self._state:
            return
        old = self._state
        if new_state in (STATUS_LOST, STATUS_DETECT):
            self._clear_tracking_resources()
        self._state = new_state
        log.info("Durum geçişi: %s → %s", old, new_state)

    def _clear_tracking_resources(self) -> None:
        if self._kalman is not None:
            self._kalman.clear()
        self._kalman = None
        self._kcf.reset()
        self._fallback_count = 0

    def _clear_kalman(self) -> None:
        if self._kalman is not None:
            self._kalman.clear()
        self._kalman = None

    def _init_kalman(self, bbox: tuple[int, int, int, int]) -> None:
        self._clear_kalman()
        k = KalmanPredictor()
        if k.init_from_bbox(bbox):
            self._kalman = k
        else:
            self._kalman = None

    def _reinit_kcf(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int],
    ) -> bool:
        self._kcf.reset()
        return self._kcf.init(frame, bbox)

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

    def _yolo_assignment(
        self,
        frame: np.ndarray,
        shared: list[Detection] | None,
        assigned: Detection | None,
    ) -> Detection | None:
        """Bu hedef için YOLO tespit eşleşmesi (MOT: ``assigned``; tekil: en iyi tespit)."""
        if assigned is not None:
            return assigned
        if not self._pooled:
            dets = self._dets(frame, shared)
            return self._best_detection(dets) if dets else None
        return None

    def _output_smooth(
        self,
        frame: np.ndarray,
        fallback_bb: tuple[int, int, int, int],
        latency_ms: float = 0.0,
    ) -> tuple[int, int, int, int]:
        if self._kalman is not None:
            if latency_ms > 0.0:
                ob = self._kalman.predict_forward(
                    latency_ms, frame.shape, self._target_fps,
                )
            else:
                ob = self._kalman.output_bbox(frame.shape)
            if ob is not None:
                return ob
        return fallback_bb

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
        latency_ms: float = 0.0,
    ) -> HybridFrameResult:
        self._pooled = shared_detections is not None
        if self._state == STATUS_DETECT:
            return self._step_detecting(
                frame, target_id, shared_detections, assigned, camera_id, latency_ms,
            )
        if self._state == STATUS_TRACK:
            return self._step_tracking(
                frame, target_id, shared_detections, assigned, camera_id, latency_ms,
            )
        return self._step_lost(
            frame, target_id, shared_detections, assigned, camera_id, latency_ms,
        )

    def _step_detecting(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
        latency_ms: float = 0.0,
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
        if not self._reinit_kcf(frame, bbox):
            log.warning("KCF init başarısız (DETECTING). bbox=%s", bbox)
            return self._fr(target_id, (0, 0, 0, 0), STATUS_DETECT, 0.0, camera_id)
        self._transition(STATUS_TRACK)
        self._track_frame_count = 0
        self._lost_frame_count = 0
        self._fallback_count = 0
        self._init_kalman(bbox)
        out_b0 = self._output_smooth(frame, bbox, latency_ms)
        return self._fr(
            target_id, out_b0, STATUS_TRACK, float(det.confidence), camera_id,
        )

    def _step_tracking(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
        latency_ms: float = 0.0,
    ) -> HybridFrameResult:
        self._track_frame_count += 1
        if self._kalman is not None:
            try:
                self._kalman.predict()
            except Exception as e:  # noqa: BLE001
                log.debug("Kalman predict: %s", e)

        yolo = self._yolo_assignment(frame, shared, assigned)

        if yolo is not None:
            bb = self._clip_bbox(yolo.bbox, frame.shape)
            if bb is not None:
                if self._kalman is not None:
                    self._kalman.correct(bb)
                self._fallback_count = 0
                if not self._reinit_kcf(frame, bb):
                    log.debug(
                        "KCF re-init (YOLO) başarısız, Kalman sürer: bbox=%s", bb,
                    )
                out_bb = self._output_smooth(frame, bb, latency_ms)
                return self._fr(
                    target_id, out_bb, STATUS_TRACK, float(yolo.confidence), camera_id,
                )
            # Geçersiz kırpım → aşağıda KCF kurtarma (YOLO yok gibi)
        if self._fallback_count < self._max_fallback_frames:
            tr: TrackResult = self._kcf.update(frame)
            if tr.success and self._kalman is not None:
                cbb = self._clip_bbox(tr.bbox, frame.shape)
                if cbb is not None:
                    self._kalman.correct(cbb)
            self._fallback_count += 1
            out_bb = self._output_smooth(frame, (0, 0, 0, 0), latency_ms)
            return self._fr(
                target_id, out_bb, STATUS_TRACK, 1.0, camera_id,
            )
        self._transition(STATUS_LOST)
        self._lost_frame_count = 0
        return self._fr(target_id, (0, 0, 0, 0), STATUS_LOST, 0.0, camera_id)

    def _step_lost(
        self,
        frame: np.ndarray,
        target_id: str,
        shared: list[Detection] | None,
        assigned: Detection | None,
        camera_id: int,
        latency_ms: float = 0.0,
    ) -> HybridFrameResult:
        dets = self._dets(frame, shared)
        reacquire: Detection | None
        if self._pooled:
            reacquire = assigned
        else:
            reacquire = self._best_detection(dets) if dets else None
        if reacquire is not None:
            bbox = self._clip_bbox(reacquire.bbox, frame.shape)
            if bbox is not None and self._reinit_kcf(frame, bbox):
                self._transition(STATUS_TRACK)
                self._track_frame_count = 0
                self._lost_frame_count = 0
                self._fallback_count = 0
                self._init_kalman(bbox)
                out_b = self._output_smooth(frame, bbox, latency_ms)
                return self._fr(
                    target_id,
                    out_b,
                    STATUS_TRACK,
                    float(reacquire.confidence),
                    camera_id,
                )
            log.warning("KCF init başarısız (LOST). bbox=%s", bbox)
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
        self._clear_tracking_resources()
        self._state = STATUS_DETECT
        self._track_frame_count = 0
        self._lost_frame_count = 0
        self._pooled = False
