"""
core/interfaces paketi — Dışarıya açılan tek import noktası.

Kullanım (projede her yerde):
    from core.interfaces import IDetector, Detection
    from core.interfaces import ITracker, TrackResult
    from core.interfaces import IStream, StreamInfo
"""

from .idetector import IDetector, Detection
from .itracker import ITracker, TrackResult
from .istream import IStream, StreamInfo

__all__ = [
    "IDetector", "Detection",
    "ITracker", "TrackResult",
    "IStream", "StreamInfo",
]
