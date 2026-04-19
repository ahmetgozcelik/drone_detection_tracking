"""
test_detector.py — ONNX model hızlı doğrulama scripti.
Model eğitimi bittikten sonra çalıştırın; test sonrası silebilirsiniz.
"""

import cv2
import numpy as np
from core.detectors.yolo_onnx import YoloOnnxDetector

detector = YoloOnnxDetector()
detector.load()

frame = cv2.imread("data/samples/test_frame.jpg")
if frame is None:
    print("[UYARI] test_frame.jpg bulunamadı — siyah kare kullanılıyor.")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

detections = detector.detect(frame)
print(f"Tespit sayısı: {len(detections)}")
for d in detections:
    print(f"  bbox={d.bbox}, conf={d.confidence:.3f}, label={d.label}")

detector.release()
