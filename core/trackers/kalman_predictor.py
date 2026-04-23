"""
kalman_predictor.py — 2B konum hız (CV) modeli: ``cv2.KalmanFilter`` sarmalayıcısı.

Sorumluluk: merkez (x, y) ve hız (dx, dy) durumu; ölçüm yalnızca merkez.
Hibrit asenkron takip çıktısında gecikmeyi azaltmak için hareket kestirimi.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

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


def _as_float_pair(cx: object, cy: object) -> tuple[float, float] | None:
    if not _is_finite_scalar(cx) or not _is_finite_scalar(cy):
        return None
    return float(cx), float(cy)


def _safe_bbox_from_center(
    cx: float, cy: float, w: int, h: int, fw: int, fh: int,
) -> tuple[int, int, int, int] | None:
    if w < 1 or h < 1 or not math.isfinite(cx) or not math.isfinite(cy):
        return None
    xi = int(round(cx - w / 2.0))
    yi = int(round(cy - h / 2.0))
    xi = max(0, min(xi, max(0, fw - 1)))
    yi = max(0, min(yi, max(0, fh - 1)))
    w2 = min(w, fw - xi)
    h2 = min(h, fh - yi)
    if w2 < 4 or h2 < 4:
        return None
    return (xi, yi, w2, h2)


class KalmanPredictor:
    """
    Durum: [cx, cy, vx, vy] — ölçüm: [cx, cy].

    ``predict()`` önce, ölçüm varsa ``correct()`` sonra çağrılmalıdır.
    """

    def __init__(self) -> None:
        self._kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        # dt = 1 kare: x' = x + vx, y' = y + vy
        f = np.eye(4, dtype=np.float32)
        f[0, 2] = 1.0
        f[1, 3] = 1.0
        self._kf.transitionMatrix = f
        h_mat = np.zeros((2, 4), dtype=np.float32)
        h_mat[0, 0] = 1.0
        h_mat[1, 1] = 1.0
        self._kf.measurementMatrix = h_mat
        # Ayarlanabilir gürültü (settings dışı varsayılanlar)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        self._w = 0
        self._h = 0
        self._frame_used_correction = False
        self._pred_cx: float = 0.0
        self._pred_cy: float = 0.0
        self._inited: bool = False

    @staticmethod
    def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def init_from_bbox(self, bbox: tuple[int, int, int, int]) -> bool:
        """İlk tespit / takip başlangıcında filtre durumunu merkez ve sıfır hıza kilitler."""
        self._frame_used_correction = False
        x, y, w, h = bbox
        if w < 4 or h < 4:
            return False
        self._w, self._h = int(w), int(h)
        cx, cy = self._center_from_bbox(bbox)
        if not (math.isfinite(cx) and math.isfinite(cy)):
            return False
        self._kf.statePost = np.array(
            [[cx], [cy], [0.0], [0.0]], dtype=np.float32,
        )
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.2
        self._inited = True
        self._pred_cx, self._pred_cy = float(cx), float(cy)
        return True

    def _clear_internals(self) -> None:
        self._frame_used_correction = False
        self._inited = False

    def clear(self) -> None:
        """Bellek: OpenCV nesnesi ve boyut state'i sıfırlanır."""
        try:
            self._kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
            f = np.eye(4, dtype=np.float32)
            f[0, 2] = 1.0
            f[1, 3] = 1.0
            self._kf.transitionMatrix = f
            h_mat = np.zeros((2, 4), dtype=np.float32)
            h_mat[0, 0] = 1.0
            h_mat[1, 1] = 1.0
            self._kf.measurementMatrix = h_mat
            self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
            self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
            self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.5
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman filtre yeniden oluşturma: %s", exc)
        self._w = 0
        self._h = 0
        self._clear_internals()

    def set_size(self, w: int, h: int) -> None:
        if w >= 4 and h >= 4:
            self._w, self._h = w, h

    def get_size(self) -> tuple[int, int]:
        return (max(self._w, 4), max(self._h, 4))

    def predict(self) -> tuple[float, float] | None:
        """
        Bir karelik ön tahmin. ``correct`` çağrılmadan önce tekrar ``predict`` çağırmayın.
        """
        if not self._inited:
            return None
        self._frame_used_correction = False
        try:
            p = self._kf.predict()
            if p is None or p.shape[0] < 2:
                return None
            c0, c1 = float(p[0, 0]), float(p[1, 0])
            if not (math.isfinite(c0) and math.isfinite(c1)):
                return None
            self._pred_cx, self._pred_cy = c0, c1
            return (c0, c1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Kalman predict: %s", exc)
            return None

    def correct(self, center_x: object, center_y: object) -> bool:
        """YOLO/CSRT merkez ölçümüyle durumu günceller."""
        pair = _as_float_pair(center_x, center_y)
        if pair is None:
            return False
        cx, cy = pair
        if not self._inited:
            return False
        try:
            z = np.array([[cx], [cy]], dtype=np.float32)
            if not np.isfinite(z).all():
                return False
            self._kf.correct(z)
            if self._kf.statePost is not None and np.isfinite(self._kf.statePost).all():
                self._frame_used_correction = True
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman correct: %s", exc)
            return False

    def output_center(self) -> tuple[float, float] | None:
        """
        Son ``predict`` / ``correct`` sonrası pürüzsüz merkez.
        Düzeltme yapıldıysa posterior; aksi halde yalnızca tahmin.
        """
        try:
            if self._frame_used_correction and self._kf.statePost is not None:
                s = self._kf.statePost
                c0, c1 = float(s[0, 0]), float(s[1, 0])
            else:
                c0, c1 = self._pred_cx, self._pred_cy
            if not (math.isfinite(c0) and math.isfinite(c1)):
                return None
            return (c0, c1)
        except Exception as exc:  # noqa: BLE001
            log.debug("Kalman output_center: %s", exc)
            return None

    def output_bbox(
        self,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        """Merkez + son bilinen w,h → bbox (kırpılmış)."""
        c = self.output_center()
        if c is None:
            return None
        cx, cy = c
        w, h = self.get_size()
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        return _safe_bbox_from_center(cx, cy, w, h, fw, fh)
