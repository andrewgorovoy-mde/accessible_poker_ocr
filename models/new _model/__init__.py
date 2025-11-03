"""
New Playing Card Detection Model

This package contains the new playing card detection system
using computer vision and edge detection techniques.
Supports both image files and live camera feed.
"""

from .card_edge_detector import CardEdgeDetector, detect_card_from_image

__all__ = ['CardEdgeDetector', 'detect_card_from_image']

