"""
Objectron NeRF Dataset Processor Utilities

This package contains utility modules for processing Objectron dataset:
- downloader: Download samples from Google Cloud Storage
- frame_extractor: Extract video frames using FFmpeg
- pose_converter: Convert ARKit poses to NeRF format
- annotation_parser: Parse protobuf annotations and extract bounding boxes
- scene_builder: Assemble final NeRF scene structure
"""

from .downloader import ObjectronDownloader
from .frame_extractor import FrameExtractor
from .pose_converter import PoseConverter
from .annotation_parser import AnnotationParser
from .scene_builder import SceneBuilder

__all__ = [
    'ObjectronDownloader',
    'FrameExtractor', 
    'PoseConverter',
    'AnnotationParser',
    'SceneBuilder'
]

__version__ = '1.0.0'