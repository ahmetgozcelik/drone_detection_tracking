"""
YOLO ONNX Runtime tabanlı IDetector implementasyonu.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from configs import settings
from configs.constants import DRONE_CLASS_ID, DRONE_LABEL, ROOT_DIR
from core.interfaces.idetector import Detection, IDetector
from utils.logger import get_logger

log = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


class YoloOnnxDetector(IDetector):
    """ONNX Runtime ile YOLO çıkarımı; model yolu ve eşikler settings.yaml üzerinden."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._model_path = self._resolve_model_path(settings["model"]["onnx_path"])
        self._conf_threshold: float = float(settings["model"]["conf_threshold"])
        self._iou_threshold: float = float(settings["model"]["iou_threshold"])
        self._input_size: int = int(settings["model"]["input_size"])

    @staticmethod
    def _resolve_model_path(path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = ROOT_DIR / p
        return p.resolve()

    def load(self) -> None:
        if not self._model_path.is_file():
            raise FileNotFoundError(f"ONNX model bulunamadı: {self._model_path}")

        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(
                settings.get("performance", {}).get("inference_threads", 2)
            )
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=so,
                providers=providers,
            )
        except Exception as e:
            raise RuntimeError(f"ONNX oturumu açılamadı: {self._model_path}") from e

        inp = self._session.get_inputs()
        if not inp:
            raise RuntimeError("ONNX model giriş tensörü tanımsız.")
        self._input_name = inp[0].name
        log.info(
            "ONNX model yüklendi: %s | input_size=%s conf=%.2f",
            self._model_path,
            self._input_size,
            self._conf_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._session is None or self._input_name is None:
            raise RuntimeError("Önce load() çağrılmalı.")

        h0, w0 = frame.shape[:2]
        blob, pad_left, pad_top, letterbox_scale = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]
        return self._postprocess(raw, w0, h0, pad_left, pad_top, letterbox_scale)

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, int, int, float]:
        """En-boy oranını koruyarak input_size kare tuval; gri (114) dolgu."""
        h0, w0 = frame.shape[:2]
        s = self._input_size
        letterbox_scale = min(s / w0, s / h0)
        new_w = int(round(w0 * letterbox_scale))
        new_h = int(round(h0 * letterbox_scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        pad_left = (s - new_w) // 2
        pad_top = (s - new_h) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, pad_left, pad_top, letterbox_scale

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, int, int, float]:
        """BGR → letterbox canvas, sonra NCHW float32 [0,1]."""
        img, pad_left, pad_top, letterbox_scale = self._letterbox(frame)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        return x, pad_left, pad_top, letterbox_scale

    def _postprocess(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        pad_left: int,
        pad_top: int,
        letterbox_scale: float,
    ) -> list[Detection]:
        """YOLOv8 ONNX çıktısı: (1, 4+nc, N) veya (1, N, 4+nc)."""
        if raw.ndim != 3:
            log.warning("Beklenmeyen çıktı boyutu: %s", raw.shape)
            return []

        b = raw[0]
        if b.shape[0] < b.shape[1]:
            pred = b.T
        else:
            pred = b

        num_cols = pred.shape[1]
        if num_cols < 5:
            return []

        xywh = pred[:, :4].astype(np.float32)
        cls_raw = pred[:, 4:]

        if cls_raw.shape[1] == 1:
            conf = _sigmoid(cls_raw[:, 0])
            class_ids = np.zeros(len(conf), dtype=np.int32)
        else:
            cls_prob = _sigmoid(cls_raw)
            class_ids = cls_prob.argmax(axis=1)
            conf = cls_prob.max(axis=1)

        mask = (conf >= self._conf_threshold) & (class_ids == DRONE_CLASS_ID)
        if not np.any(mask):
            return []

        xywh = xywh[mask]
        conf = conf[mask]

        cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        boxes_px = np.stack([x1, y1, bw, bh], axis=1)

        boxes_px[:, 0] -= pad_left
        boxes_px[:, 1] -= pad_top
        boxes_px /= letterbox_scale

        boxes_px[:, 0] = np.clip(boxes_px[:, 0], 0, orig_w - 1)
        boxes_px[:, 1] = np.clip(boxes_px[:, 1], 0, orig_h - 1)
        boxes_px[:, 2] = np.clip(boxes_px[:, 2], 1, orig_w)
        boxes_px[:, 3] = np.clip(boxes_px[:, 3], 1, orig_h)

        boxes_list = boxes_px.tolist()
        scores_list = conf.tolist()
        try:
            idx = cv2.dnn.NMSBoxes(
                boxes_list,
                scores_list,
                self._conf_threshold,
                self._iou_threshold,
            )
        except cv2.error:
            idx = ()

        if idx is None or len(idx) == 0:
            return []

        if isinstance(idx, np.ndarray):
            flat = idx.flatten().tolist()
        else:
            flat = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idx]

        out: list[Detection] = []
        for i in flat:
            x, y, w, h = boxes_px[i]
            out.append(
                Detection(
                    bbox=(int(round(x)), int(round(y)), int(round(w)), int(round(h))),
                    confidence=float(conf[i]),
                    class_id=DRONE_CLASS_ID,
                    label=DRONE_LABEL,
                )
            )
        return out

    def release(self) -> None:
        self._session = None
        self._input_name = None
        log.debug("ONNX oturumu serbest bırakıldı.")
