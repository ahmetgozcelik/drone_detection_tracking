"""
kalman_predictor.py — 8B sabit hız (CV) bbox modeli: ``cv2.KalmanFilter(8, 4)``.

Durum: [cx, cy, w, h, dx, dy, dw, dh] (merkez + boyut) — ölçüm: [cx, cy, w, h].
YOLO / KCF sol-üst (x, y, w, h) ölçümü; içeride merkez birimine çevrilir.
Dışa dönen ``output_bbox`` / ``predict_forward`` sol-üst OpenCV kutu (geri uyum).

Hibrit asenkron takip: hız sınırlandırma (kutu sapan / divergence azaltma),
``predict_forward`` ile gecikme telafili ileri kestirim.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from configs.constants import KALMAN_CLAMP_DWH_PX, KALMAN_CLAMP_DXY_PX
from utils.logger import get_logger

log = get_logger(__name__)


def _is_finite_scalar(v: object) -> bool:
    if v is None or isinstance(v, bool):
        return False
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(x))


def _as_finite_bbox(
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x, y, w, h = bbox
    if not all(_is_finite_scalar(v) for v in (x, y, w, h)):
        return None
    xi, yi, wi, hi = int(x), int(y), int(w), int(h)
    if wi < 4 or hi < 4:
        return None
    return (xi, yi, wi, hi)


def _tl_to_center(
    bbox: tuple[int, int, int, int],
) -> tuple[float, float, int, int] | None:
    t = _as_finite_bbox(bbox)
    if t is None:
        return None
    x, y, w, h = t
    return (x + w / 2.0, y + h / 2.0, w, h)


def _safe_bbox_tl(
    x_tl: float, y_tl: float, w: float, h: float, fw: int, fh: int,
) -> tuple[int, int, int, int] | None:
    if not all(math.isfinite(v) for v in (x_tl, y_tl, w, h)):
        return None
    if w < 1 or h < 1:
        return None
    xi = int(round(x_tl))
    yi = int(round(y_tl))
    wi = int(round(w))
    hi = int(round(h))
    xi = max(0, min(xi, max(0, fw - 1)))
    yi = max(0, min(yi, max(0, fh - 1)))
    w2 = min(wi, max(0, fw - xi))
    h2 = min(hi, max(0, fh - yi))
    if w2 < 4 or h2 < 4:
        return None
    return (xi, yi, w2, h2)


def _center_to_tl(
    cx: float, cy: float, w: float, h: float, fw: int, fh: int,
) -> tuple[int, int, int, int] | None:
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return None
    if w < 1 or h < 1:
        return None
    x_tl = cx - w / 2.0
    y_tl = cy - h / 2.0
    return _safe_bbox_tl(x_tl, y_tl, w, h, fw, fh)


class KalmanPredictor:
    """
    Durum: [cx, cy, w, h, dx, dy, dw, dh] — ölçüm: merkez [cx, cy, w, h].

    ``predict()`` önce, ölçüm varsa ``correct(bbox_tl)`` sonra çağrılmalıdır.
    ``correct`` YOLO sol-üst (x, y, w, h) kabul eder, içeride merkeze çevrilir.
    """

    @staticmethod
    def _build_transition_matrix(dt: float) -> np.ndarray:
        f = np.eye(8, dtype=np.float32)
        dtf = float(dt)
        for i in range(4):
            f[i, i + 4] = dtf
        return f

    def __init__(self) -> None:
        self._kf = cv2.KalmanFilter(8, 4, 0, cv2.CV_32F)
        self._kf.transitionMatrix = self._build_transition_matrix(1.0)
        h_mat = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            h_mat[i, i] = 1.0
        self._kf.measurementMatrix = h_mat
        p_noise = np.eye(8, dtype=np.float32)
        p_noise[0, 0] = p_noise[1, 1] = 1e-1
        p_noise[2, 2] = p_noise[3, 3] = 1e-1
        p_noise[4, 4] = p_noise[5, 5] = 5.0
        p_noise[6, 6] = p_noise[7, 7] = 5.0
        self._kf.processNoiseCov = p_noise
        self._kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self._kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.5
        self._frame_used_correction = False
        self._pred_cxywh: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._inited: bool = False

    def _clamp_state_velocities(self) -> None:
        for attr in ("statePre", "statePost"):
            s = getattr(self._kf, attr, None)
            if s is None or s.shape[0] < 8:
                continue
            s[4, 0] = np.clip(
                s[4, 0], -KALMAN_CLAMP_DXY_PX, KALMAN_CLAMP_DXY_PX,
            )
            s[5, 0] = np.clip(
                s[5, 0], -KALMAN_CLAMP_DXY_PX, KALMAN_CLAMP_DXY_PX,
            )
            s[6, 0] = np.clip(
                s[6, 0], -KALMAN_CLAMP_DWH_PX, KALMAN_CLAMP_DWH_PX,
            )
            s[7, 0] = np.clip(
                s[7, 0], -KALMAN_CLAMP_DWH_PX, KALMAN_CLAMP_DWH_PX,
            )

    def init_from_bbox(self, bbox_tl: tuple[int, int, int, int]) -> bool:
        """Sol-üst YOLO bbox; durum [cx, cy, w, h, 0,0,0,0] kilitler."""
        self._frame_used_correction = False
        cm = _tl_to_center(bbox_tl)
        if cm is None:
            return False
        cx, cy, w, h = cm[0], cm[1], float(cm[2]), float(cm[3])
        self._kf.statePost = np.array(
            [
                [float(cx)], [float(cy)], [float(w)], [float(h)],
                [0.0], [0.0], [0.0], [0.0],
            ],
            dtype=np.float32,
        )
        self._kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.2
        self._inited = True
        self._pred_cxywh = (float(cx), float(cy), float(w), float(h))
        return True

    def _clear_internals(self) -> None:
        self._frame_used_correction = False
        self._inited = False

    def clear(self) -> None:
        try:
            self._kf = cv2.KalmanFilter(8, 4, 0, cv2.CV_32F)
            self._kf.transitionMatrix = self._build_transition_matrix(1.0)
            h_mat = np.zeros((4, 8), dtype=np.float32)
            for i in range(4):
                h_mat[i, i] = 1.0
            self._kf.measurementMatrix = h_mat
            p_noise = np.eye(8, dtype=np.float32)
            p_noise[0, 0] = p_noise[1, 1] = 1e-1
            p_noise[2, 2] = p_noise[3, 3] = 1e-1
            p_noise[4, 4] = p_noise[5, 5] = 5.0
            p_noise[6, 6] = p_noise[7, 7] = 5.0
            self._kf.processNoiseCov = p_noise
            self._kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
            self._kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.5
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman filtre yeniden oluşturma: %s", exc)
        self._clear_internals()

    def predict(self, dt: float = 1.0) -> tuple[float, float, float, float] | None:
        """Bir adımlık ön tahmin; ilk dört değer (cx, cy, w, h)."""
        if not self._inited:
            return None
        self._frame_used_correction = False
        try:
            dtf = float(dt)
            if not math.isfinite(dtf) or dtf <= 0:
                dtf = 1.0
            self._kf.transitionMatrix = self._build_transition_matrix(dtf)
            p = self._kf.predict()
            self._clamp_state_velocities()
            if p is None or p.shape[0] < 4:
                return None
            cx, cy, w, h = float(p[0, 0]), float(p[1, 0]), float(p[2, 0]), float(p[3, 0])
            if not all(math.isfinite(v) for v in (cx, cy, w, h)):
                return None
            self._pred_cxywh = (cx, cy, w, h)
            return (cx, cy, w, h)
        except Exception as exc:  # noqa: BLE001
            log.warning("Kalman predict: %s", exc)
            return None

    def correct(self, bbox_tl: tuple[int, int, int, int]) -> bool:
        """Sol-üst (x, y, w, h) ölçümü; Kalman z = [cx, cy, w, h]."""
        cm = _tl_to_center(bbox_tl)
        if cm is None or not self._inited:
            return False
        cx, cy, w, h = cm[0], cm[1], float(cm[2]), float(cm[3])
        try:
            z = np.array(
                [[float(cx)], [float(cy)], [float(w)], [float(h)]],
                dtype=np.float32,
            )
            if not np.isfinite(z).all():
                return False
            self._kf.correct(z)
            if self._kf.statePost is not None and np.isfinite(self._kf.statePost).all():
                self._frame_used_correction = True
            self._clamp_state_velocities()
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman correct: %s", exc)
            return False

    def output_bbox(
        self,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        """Sol-üst (x, y, w, h) kırpılmış; motor geri kalanı OpenCV ile uyumlu tutar."""
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        try:
            if self._frame_used_correction and self._kf.statePost is not None:
                s = self._kf.statePost
                cx, cy, w, h = (
                    float(s[0, 0]), float(s[1, 0]),
                    float(s[2, 0]), float(s[3, 0]),
                )
            else:
                cx, cy, w, h = self._pred_cxywh
            return _center_to_tl(cx, cy, w, h, fw, fh)
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman output_bbox: %s", exc)
            return None

    def predict_forward(
        self,
        latency_ms: float = 30.0,
        frame_shape: tuple[int, ...] = (480, 640, 3),
        frame_fps: float = 30.0,
    ) -> tuple[int, int, int, int] | None:
        """
        Gecikme telafili ileri kestirim. ``dt_frames`` transitionMatrix (F) ile
        aynı sabit hız adımıdır; hızlar önce sınırlandırılır (divergence önlemi).
        ``frame_shape``: (H, W) veya (H, W, C) — sadece H, W okunur.
        """
        if not self._inited:
            return None
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        try:
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                return self.output_bbox(frame_shape)
            fps = float(frame_fps)
            if not math.isfinite(fps) or fps < 1.0:
                fps = 30.0
            frame_period_ms = 1000.0 / fps
            dt_frames = float(latency_ms) / frame_period_ms
            if not math.isfinite(dt_frames):
                return self.output_bbox(frame_shape)
            if self._kf.statePost is None or self._kf.statePost.shape[0] < 8:
                return self.output_bbox(frame_shape)
            self._clamp_state_velocities()
            s = self._kf.statePost
            cx, cy, w, h = (
                float(s[0, 0]), float(s[1, 0]),
                float(s[2, 0]), float(s[3, 0]),
            )
            dx, dy, dw, dh = (
                float(s[4, 0]), float(s[5, 0]),
                float(s[6, 0]), float(s[7, 0]),
            )
            if not all(
                math.isfinite(v) for v in (cx, cy, w, h, dx, dy, dw, dh)
            ):
                return self.output_bbox(frame_shape)
            # F(dt): x' = x + dt * v — transitionMatrix üzerinden dt kullanımı
            fdt = self._build_transition_matrix(dt_frames)
            cxf = cx + fdt[0, 4] * dx
            cyf = cy + fdt[1, 5] * dy
            wf = w + fdt[2, 6] * dw
            hf = h + fdt[3, 7] * dh
            if not all(math.isfinite(v) for v in (cxf, cyf, wf, hf)):
                return self.output_bbox(frame_shape)
            return _center_to_tl(cxf, cyf, wf, hf, fw, fh)
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman predict_forward: %s", exc)
            return self.output_bbox(frame_shape)
