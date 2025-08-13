"""
AutoCut v2 - Automated Video Editing Tool

A powerful video processing and automation library for cutting, trimming,
and batch processing video content.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.processor import AutoCut
from .core.scene_detector import SceneDetector
from .processors.video_cutter import VideoCutter

__all__ = [
    "AutoCut",
    "SceneDetector", 
    "VideoCutter",
]
