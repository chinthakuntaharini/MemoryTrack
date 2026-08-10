"""
Core modules for MemoryTrack system.
"""

from .detector import PersonDetector, MultiObjectTracker, BoundingBox, Track
from .pose_extractor import PoseExtractor, PoseFeatures, BodyRatioCalculator
from .reid_extractor import ReIDExtractor, SimpleReIDExtractor
from .color_extractor import ColorExtractor, ColorFeatures
from .feature_fusion import FeatureFusion, AdaptiveFeatureFusion, FusionResult
from .memory_bank import (
    AdaptiveMemoryBank,
    PersonProfile,
    TemporalSnapshot,
    MatchResult
)
from .accessory_extractor import AccessoryExtractor, AccessoryFeatures, AccessoryDetection
from .occlusion_detector import OcclusionDetector, OcclusionResult
from .xai import ExplanationGenerator, Explanation

__all__ = [
    'PersonDetector',
    'MultiObjectTracker',
    'BoundingBox',
    'Track',
    'PoseExtractor',
    'PoseFeatures',
    'BodyRatioCalculator',
    'ReIDExtractor',
    'SimpleReIDExtractor',
    'ColorExtractor',
    'ColorFeatures',
    'FeatureFusion',
    'AdaptiveFeatureFusion',
    'FusionResult',
    'AdaptiveMemoryBank',
    'PersonProfile',
    'TemporalSnapshot',
    'MatchResult',
    'AccessoryExtractor',
    'AccessoryFeatures',
    'AccessoryDetection',
    'OcclusionDetector',
    'OcclusionResult',
    'ExplanationGenerator',
    'Explanation'
]
