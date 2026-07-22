"""Day/night classification from brightness and color-temperature heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.brightness import frame_brightness
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)

BRIGHTNESS_NIGHT_THRESHOLD = 0.28
BRIGHTNESS_DAY_THRESHOLD = 0.42


class LightingCondition(str, Enum):
    DAY = "day"
    DUSK_DAWN = "dusk_dawn"
    NIGHT = "night"


def _warm_channel_ratio(frame: np.ndarray) -> float:
    """Ratio of warm (red+green artificial-light) to cool (blue) channel energy.

    Night footage with streetlights/headlights tends to skew warm relative to
    daylight's more balanced or cool-skewed spectrum.
    """
    b, g, r = (c.astype(np.float64) for c in cv2.split(frame.astype(np.float32)))
    warm = np.mean(r) + np.mean(g)
    cool = np.mean(b) + 1e-6
    return float(warm / cool)


@dataclass(frozen=True, slots=True)
class LightingClassification:
    frame_index: int
    timestamp: float
    brightness: float
    warm_ratio: float
    condition: LightingCondition


def classify_frame_lighting(frame: np.ndarray) -> tuple[LightingCondition, float, float]:
    brightness = frame_brightness(frame)
    warm_ratio = _warm_channel_ratio(frame)

    if brightness < BRIGHTNESS_NIGHT_THRESHOLD:
        condition = LightingCondition.NIGHT
    elif brightness > BRIGHTNESS_DAY_THRESHOLD:
        condition = LightingCondition.DAY
    else:
        condition = LightingCondition.DUSK_DAWN
    return condition, brightness, warm_ratio


def classify_video_lighting(
    video_path: Path, *, stride: int = 30, max_seconds: float | None = None
) -> list[LightingClassification]:
    """Classify lighting condition across sampled frames of a video."""
    results: list[LightingClassification] = []
    for index, timestamp, frame in iter_frames(video_path, stride=stride, max_seconds=max_seconds):
        condition, brightness, warm_ratio = classify_frame_lighting(frame)
        results.append(LightingClassification(index, timestamp, brightness, warm_ratio, condition))

    if results:
        dominant = max(
            {c.condition for c in results},
            key=lambda cond: sum(r.condition == cond for r in results),
        )
        logger.info(
            "day_night(%s): dominant=%s over %d samples", video_path.name, dominant.value, len(results)
        )
    return results
