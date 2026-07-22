"""Coarse heuristic scene classification (sky/foliage/road/urban) from color and texture.

This is deliberately heuristic rather than a trained model, to avoid pulling in
extra downloaded weights for a coarse signal that only needs to feed the
highlight/aesthetic scoring engine, not stand alone as a precise classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


class SceneCategory(str, Enum):
    SKY = "sky"
    FOLIAGE = "foliage"
    ROAD = "road"
    URBAN = "urban"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SceneClassification:
    category: SceneCategory
    sky_fraction: float
    foliage_fraction: float
    road_fraction: float
    edge_density: float


def classify_frame_scene(frame: np.ndarray) -> SceneClassification:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total_pixels = frame.shape[0] * frame.shape[1]

    # Sky: blue/white hues, high value, low-to-moderate saturation.
    sky_mask = ((h >= 90) & (h <= 130) & (v > 120)) | ((s < 30) & (v > 180))
    sky_fraction = float(np.count_nonzero(sky_mask)) / total_pixels

    # Foliage: green hues with reasonable saturation.
    foliage_mask = (h >= 35) & (h <= 85) & (s > 40)
    foliage_fraction = float(np.count_nonzero(foliage_mask)) / total_pixels

    # Road: low saturation, mid-to-low value (gray asphalt), concentrated
    # toward the lower half of the frame where a road/trail would be.
    lower_half = slice(frame.shape[0] // 2, frame.shape[0])
    lower_s, lower_v = s[lower_half], v[lower_half]
    road_mask = (lower_s < 40) & (lower_v > 30) & (lower_v < 160)
    road_fraction = float(np.count_nonzero(road_mask)) / (total_pixels / 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / total_pixels

    fractions = {
        SceneCategory.SKY: sky_fraction,
        SceneCategory.FOLIAGE: foliage_fraction,
        SceneCategory.ROAD: road_fraction,
    }
    dominant_category, dominant_fraction = max(fractions.items(), key=lambda kv: kv[1])

    if edge_density > 0.12 and dominant_fraction < 0.35:
        category = SceneCategory.URBAN
    elif dominant_fraction > 0.15:
        category = dominant_category
    else:
        category = SceneCategory.UNKNOWN

    return SceneClassification(
        category=category,
        sky_fraction=sky_fraction,
        foliage_fraction=foliage_fraction,
        road_fraction=road_fraction,
        edge_density=edge_density,
    )
