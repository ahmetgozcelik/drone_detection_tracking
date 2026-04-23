"""
test_composite_stream.py — CompositeStream + TaggedFrame testleri (T3).

Donanım gerektirmez. İki FileStream ile çift kamera simülasyonu yapar.
Video dosyası yoksa mock IStream kullanılır.

Çalıştırma:
    pytest tests/test_composite_stream.py -v

Kapsam:
    - TaggedFrame: meta["camera_id"] doğru taşımalı
    - TaggedFrame: numpy işlemleri bozulmamalı
    - CompositeStream: round-robin sırası (0, 1, 0, 1, ...)
    - CompositeStream: camera_id TaggedFrame meta'sında doğru
    - CompositeStream: bir kamera None döndürünce diğerine geçmeli
    - CompositeStream: tüm kameralar None → None dönmeli
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np


class TestTaggedFrame(unittest.TestCase):
    """TaggedFrame numpy alt sınıfı testleri."""

    def setUp(self):
        from infrastructure.tagged_frame import TaggedFrame
        self.TaggedFrame = TaggedFrame

    def test_from_frame_sets_camera_id(self):
        """from_frame meta['camera_id'] doğru ayarlamalı."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tagged = self.TaggedFrame.from_frame(frame, camera_id=1)
        self.assertEqual(tagged.meta["camera_id"], 1)

    def test_is_numpy_array(self):
        """TaggedFrame gerçek bir numpy dizisi olmalı."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tagged = self.TaggedFrame.from_frame(frame, camera_id=0)
        self.assertIsInstance(tagged, np.ndarray)

    def test_shape_preserved(self):
        """Frame boyutları korunmalı."""
        frame = np.ones((1080, 1920, 3), dtype=np.uint8)
        tagged = self.TaggedFrame.from_frame(frame, camera_id=0)
        self.assertEqual(tagged.shape, (1080, 1920, 3))

    def test_dtype_preserved(self):
        """Frame dtype korunmalı."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tagged = self.TaggedFrame.from_frame(frame, camera_id=0)
        self.assertEqual(tagged.dtype, np.uint8)

    def test_numpy_arithmetic_works(self):
        """Numpy işlemleri (toplama vb.) bozulmamalı."""
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 10
        tagged = self.TaggedFrame.from_frame(frame, camera_id=0)
        result = tagged + 5
        self.assertTrue(np.all(result == 15))

    def test_meta_camera_id_zero_by_default(self):
        """meta dict'te camera_id default 0 olmalı."""
        from infrastructure.tagged_frame import TaggedFrame
        tagged = TaggedFrame(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(tagged.meta.get("camera_id", 0), 0)

    def test_slicing_preserves_meta(self):
        """Dilimleme sonrası meta korunmalı."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tagged = self.TaggedFrame.from_frame(frame, camera_id=1)
        sliced = tagged[0:50, 0:50]
        self.assertEqual(sliced.meta.get("camera_id"), 1)


class MockStream:
    """Test için minimal IStream benzeri nesne."""

    def __init__(self, frames, camera_id_hint=0):
        self._frames = iter(frames)
        self._open = False
        self.camera_id_hint = camera_id_hint

    def open(self):
        self._open = True
        return True

    def read(self):
        try:
            return next(self._frames)
        except StopIteration:
            return None

    def is_open(self):
        return self._open

    def release(self):
        self._open = False

    def get_info(self):
        from core.interfaces.istream import StreamInfo
        return StreamInfo(
            source_uri=f"mock://{self.camera_id_hint}",
            width=640, height=480, fps=30.0, is_live=True,
        )


class TestCompositeStream(unittest.TestCase):
    """CompositeStream round-robin ve TaggedFrame meta testleri."""

    def _make_frame(self, val=0):
        return np.ones((480, 640, 3), dtype=np.uint8) * val

    def test_round_robin_order(self):
        """read() sırası: kamera 0, kamera 1, kamera 0, kamera 1 ..."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([self._make_frame(10), self._make_frame(10)])
        stream_b = MockStream([self._make_frame(20), self._make_frame(20)])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()

        frames = [comp.read() for _ in range(4)]
        cam_ids = [f.meta["camera_id"] for f in frames]
        self.assertEqual(cam_ids, [0, 1, 0, 1])

    def test_camera_id_in_tagged_frame(self):
        """Döndürülen frame'de meta['camera_id'] doğru kamera numarasını içermeli."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([self._make_frame(1)])
        stream_b = MockStream([self._make_frame(2)])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()

        frame_a = comp.read()
        frame_b = comp.read()
        self.assertEqual(frame_a.meta["camera_id"], 0)
        self.assertEqual(frame_b.meta["camera_id"], 1)

    def test_skips_none_stream(self):
        """Bir kamera None döndürünce diğerinden frame alınmalı."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([None])       # ilk çağrı None
        stream_b = MockStream([self._make_frame(99)])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()

        frame = comp.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.meta["camera_id"], 1)

    def test_all_none_returns_none(self):
        """Tüm kameralar None dönünce CompositeStream None dönmeli."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([None])
        stream_b = MockStream([None])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()

        frame = comp.read()
        self.assertIsNone(frame)

    def test_open_opens_all_streams(self):
        """open() tüm alt stream'leri açmalı."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([])
        stream_b = MockStream([])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()
        self.assertTrue(stream_a.is_open())
        self.assertTrue(stream_b.is_open())

    def test_release_closes_all_streams(self):
        """release() tüm alt stream'leri kapatmalı."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([])
        stream_b = MockStream([])
        comp = CompositeStream([stream_a, stream_b])
        comp.open()
        comp.release()
        self.assertFalse(stream_a.is_open())
        self.assertFalse(stream_b.is_open())

    def test_empty_streams_raises(self):
        """Boş stream listesi ValueError fırlatmalı."""
        from infrastructure.streams.composite_stream import CompositeStream
        with self.assertRaises(ValueError):
            CompositeStream([])

    def test_is_open_true_if_any_open(self):
        """En az bir stream açıksa is_open() True olmalı."""
        from infrastructure.streams.composite_stream import CompositeStream

        stream_a = MockStream([])
        stream_b = MockStream([])
        comp = CompositeStream([stream_a, stream_b])
        stream_a.open()   # sadece A
        self.assertTrue(comp.is_open())


if __name__ == "__main__":
    unittest.main(verbosity=2)
