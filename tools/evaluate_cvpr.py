#!/usr/bin/env python3
"""
evaluate_cvpr.py — CVPR Anti-UAV tarzı etiketlerle hibrit (YOLO+tracker) çevrimdışı ölçüm.

UI açılmaz; SystemController (hibrit asenkron takip ile aynı process() yolu) kullanılır.
Beklenen etiket biçimleri:
  * TXT: satır başına `frame_id x y w h` (piksel, boşluk veya virgül; 0 tabanlı kare
    indeksi). `#` ile başlayan ve boş satırlar yok sayılır.
  * JSON: { "0": [x,y,w,h], "1": [...] } veya { "frames": { "0": [...] } } veya
    [ {"frame": 0, "bbox": [x,y,w,h]}, ... ]

mAP@0.5: tek sınıf / tek tespit varsayımı — GT var olan karelerde IoU >= 0.5 oranı
(COCO mAP(0.5) ile uyumlu basit sınıf AP).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Proje kökü
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from core.engine.controller import SystemController
from core.trackers.hybrid_tracker import HybridFrameResult
from utils.logger import get_logger

log = get_logger(__name__)

EMPTY_BBOX: tuple[int, int, int, int] = (0, 0, 0, 0)


def _intersection_over_union(
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


def _load_labels_json(path: Path) -> dict[int, tuple[int, int, int, int]]:
    raw = path.read_text(encoding="utf-8")
    data: Any = json.loads(raw)
    out: dict[int, tuple[int, int, int, int]] = {}

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            fi = int(item.get("frame", item.get("index", 0)))
            box = item.get("bbox", item.get("box", [0, 0, 0, 0]))
            if len(box) >= 4:
                out[fi] = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return out

    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                fi = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, (list, tuple)) or len(v) < 4:
                continue
            out[fi] = (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    return out


def _load_labels_txt(path: Path) -> dict[int, tuple[int, int, int, int]]:
    out: dict[int, tuple[int, int, int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            fi = int(float(parts[0]))
            x, y, w, h = (int(float(parts[1])), int(float(parts[2])),
                          int(float(parts[3])), int(float(parts[4])))
        except ValueError:
            continue
        out[fi] = (x, y, w, h)
    return out


def load_labels(path: Path) -> dict[int, tuple[int, int, int, int]]:
    p = path.resolve()
    if p.suffix.lower() == ".json":
        return _load_labels_json(p)
    return _load_labels_txt(p)


def _pred_bbox(r: HybridFrameResult) -> tuple[int, int, int, int]:
    b = r.bbox
    if b == (0, 0, 0, 0):
        return EMPTY_BBOX
    return b


def iter_video_frames(
    cap: cv2.VideoCapture, max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        yield idx, bgr
        idx += 1


def run_offline_eval(
    video: Path, labels: dict[int, tuple[int, int, int, int]],
) -> tuple[float, float, int, int, list[float]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Video açılamadı: {video}")

    ctrl = SystemController.build_default()
    ctrl.load()

    ious: list[float] = []
    tp_50 = 0
    n_gt = 0

    try:
        for frame_idx, bgr in iter_video_frames(cap):
            gt = labels.get(frame_idx)
            if gt is None or (gt[2] < 1 or gt[3] < 1):
                # Etiket yok veya boş: bu kareyi mAP/IoU dışı bırak
                # Hibrit durum yine ilerlesin
                _ = ctrl.process(bgr)
                continue

            n_gt += 1
            result = ctrl.process(bgr)

            pred = _pred_bbox(result)
            if pred == EMPTY_BBOX:
                iou = 0.0
            else:
                iou = _intersection_over_union(pred, gt)
            ious.append(iou)
            if iou >= 0.5:
                tp_50 += 1
    finally:
        cap.release()
        ctrl.release()

    n_eval = len(ious)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    map_50 = (tp_50 / n_gt) if n_gt > 0 else 0.0
    return mean_iou, map_50, n_gt, n_eval, ious


def _print_report(
    video: Path,
    label_path: Path,
    mean_iou: float,
    map_50: float,
    n_eval_frames: int,
    ious: list[float],
) -> None:
    print()
    print("=" * 64)
    print("  Hibrit takip — çevrimdışı değerlendirme (inference, UI yok)")
    print("=" * 64)
    print(f"  Video          : {video}")
    print(f"  Etiketler     : {label_path}")
    print(f"  GT'li kare     : {n_eval_frames}")
    print(f"  mAP@0.5        : {map_50:.4f}  (IoU eşiği 0,50)")
    print(f"  Ortalama IoU   : {mean_iou:.4f}")
    if ious:
        print(f"  Medyan IoU     : {float(np.median(ious)):.4f}")
    print("=" * 64)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CVPR-style Anti-UAV etiketleriyle hibrit tracker ölçümü",
    )
    ap.add_argument("--video", type=Path, required=True, help="MP4 / AVI yolu")
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="txt veya json etiket dosyası",
    )
    args = ap.parse_args()
    v = args.video.resolve()
    lab = args.labels.resolve()
    if not v.is_file():
        log.error("Video bulunamadı: %s", v)
        return 1
    if not lab.is_file():
        log.error("Etiket dosyası yok: %s", lab)
        return 1

    d = load_labels(lab)
    if not d:
        log.error("Okunan etiket sözlüğü boş. Biçim: frame x y w h veya json.")
        return 1

    mean_iou, map_50, n_gt, n_eval, ious = run_offline_eval(v, d)
    _print_report(v, lab, mean_iou, map_50, n_eval, ious)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())