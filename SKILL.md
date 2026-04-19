# Drone Tracking System — Proje Skill Dosyası
# Bu dosya Claude ve Cursor'un projeyi tutarlı anlaması için yazılmıştır.
# Konum: SKILL.md (proje kökünde)

## Proje Özeti

Raspberry Pi 4 / Jetson Nano hedefli, gerçek zamanlı drone tespit ve takip sistemi.
YOLOv8n ONNX + OpenCV CSRT/KCF hibrit mimarisi. PyQt5 arayüzü.

## Dosya Haritası ve Sorumluluklar

| Dosya | Sorumluluk | Bağımlı Olduğu |
|-------|-----------|----------------|
| `core/interfaces/idetector.py` | Detection sözleşmesi | — |
| `core/interfaces/itracker.py` | TrackResult sözleşmesi | — |
| `core/interfaces/istream.py` | Stream sözleşmesi | — |
| `core/detectors/yolo_onnx.py` | ONNX çıkarımı + letterbox | IDetector, settings |
| `core/trackers/base_trackers.py` | CSRT/KCF wrappers | ITracker |
| `core/trackers/hybrid_tracker.py` | Durum makinesi | IDetector, ITracker |
| `core/engine/controller.py` | Detector+Tracker yönetimi | HybridTracker |
| `core/engine/pipeline.py` | 2 thread orkestrasyon | Controller, StreamManager |
| `infrastructure/stream_manager.py` | Capture thread + reconnect | IStream |
| `utils/logger.py` | Merkezi loglama | settings |
| `utils/metrics.py` | FPS/latency/RAM ölçümü | settings |
| `configs/constants.py` | Değişmez sabitler | — |
| `configs/settings.yaml` | Runtime ayarlar | — |
| `ui/main_window.py` | PyQt5 pencere | Pipeline, constants |

## Kritik Veri Akışı

```
Kamera/Dosya
    ↓
IStream.read()
    ↓
StreamManager (Thread 1: Capture)
    ↓ Queue(maxsize=2)
Pipeline._inference_loop (Thread 2: Inference)
    ↓
SystemController.process(frame)
    ↓
HybridTracker.process(frame)
    ↓ [DETECTING]          ↓ [TRACKING]
IDetector.detect()    ITracker.update()
    ↓
HybridFrameResult(bbox, status)
    ↓
PerformanceMonitor.snapshot() → MetricsSnapshot
    ↓ pyqtSignal (thread-safe)
MainWindow._update_display()
```

## Durum Makinesi

```
DETECTING ──tespit varsa──→ TRACKING
TRACKING ──takip kayıpsa──→ LOST
TRACKING ──periyodik YOLO, tespit yoksa──→ LOST
LOST ──tespit varsa──→ TRACKING
LOST ──redetect_interval karede bulunamazsa──→ DETECTING
```

## Edge Device Kısıtları (Değiştirilemez)

- Max RAM: 2 GB (settings: max_ram_mb = 1800)
- Max thread: 3 (Capture + Inference + Qt main)
- Sadece ONNX Runtime — PyTorch import edilmez
- Model: YOLOv8n nano — başka mimari kullanılamaz
- GC: Her 100 karede gc.collect()

## Sık Sorulan Sorular

**Q: Yeni bir kamera kaynağı nasıl eklenir?**
A: `IStream` arayüzünü implemente eden yeni bir sınıf yaz. `pipeline.py` ve `stream_manager.py` değişmez.

**Q: Farklı bir detector denenecekse?**
A: `IDetector`'ı implemente et, `SystemController.build_default()` içine yeni bir factory branch ekle.

**Q: CSRT yerine KCF kullanmak için?**
A: `settings.yaml` → `tracker.mode: "kcf"` — kod değişikliği gerekmez.

**Q: Confidence eşiği nasıl değiştirilir?**
A: `settings.yaml` → `model.conf_threshold` — kod değişikliği gerekmez.
