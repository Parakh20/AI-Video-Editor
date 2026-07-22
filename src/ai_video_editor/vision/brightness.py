"""Per-frame and per-clip brightness estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)


def frame_brightness(frame: np.ndarray) -> float:
    """Mean luminance of a BGR frame, normalized to [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(np.mean(gray)) / 255.0


@dataclass(frozen=True, slots=True)
class BrightnessSample:
    frame_index: int
    timestamp: float
    brightness: float


@dataclass(frozen=True, slots=True)
class BrightnessReport:
    samples: list[BrightnessSample]
    mean_brightness: float
    min_brightness: float
    max_brightness: float


def analyze_video_brightness(
    video_path: Path, *, stride: int = 15, max_seconds: float | None = None
) -> BrightnessReport:
    """Sample brightness across a video at the given frame stride."""
    samples: list[BrightnessSample] = []
    for index, timestamp, frame in iter_frames(video_path, stride=stride, max_seconds=max_seconds):
        samples.append(BrightnessSample(index, timestamp, frame_brightness(frame)))

    if not samples:
        raise ValueError(f"No frames read from {video_path}")

    values = [s.brightness for s in samples]
    report = BrightnessReport(
        samples=samples,
        mean_brightness=float(np.mean(values)),
        min_brightness=float(np.min(values)),
        max_brightness=float(np.max(values)),
    )
    logger.info(
        "brightness(%s): mean=%.3f min=%.3f max=%.3f over %d samples",
        video_path.name,
        report.mean_brightness,
        report.min_brightness,
        report.max_brightness,
        len(samples),
    )
    return report
