"""Laplacian-variance based blur detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)

# Below this Laplacian variance, a frame is considered blurry. Tuned for
# 1080p-2.7K action-cam footage; empirically sharp handheld/drone frames
# sit well above 100, motion-blurred or out-of-focus frames sit below 60.
DEFAULT_BLUR_THRESHOLD = 60.0


def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of the Laplacian of a frame — higher means sharper edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(frozen=True, slots=True)
class BlurSample:
    frame_index: int
    timestamp: float
    laplacian_variance: float
    is_blurry: bool


@dataclass(frozen=True, slots=True)
class BlurReport:
    samples: list[BlurSample]
    blurry_fraction: float
    mean_variance: float


def analyze_video_blur(
    video_path: Path,
    *,
    stride: int = 15,
    max_seconds: float | None = None,
    threshold: float = DEFAULT_BLUR_THRESHOLD,
) -> BlurReport:
    """Sample blur across a video and report the fraction of blurry frames."""
    samples: list[BlurSample] = []
    for index, timestamp, frame in iter_frames(video_path, stride=stride, max_seconds=max_seconds):
        variance = laplacian_variance(frame)
        samples.append(BlurSample(index, timestamp, variance, variance < threshold))

    if not samples:
        raise ValueError(f"No frames read from {video_path}")

    variances = [s.laplacian_variance for s in samples]
    blurry_fraction = sum(s.is_blurry for s in samples) / len(samples)
    report = BlurReport(
        samples=samples, blurry_fraction=blurry_fraction, mean_variance=float(np.mean(variances))
    )
    logger.info(
        "blur(%s): blurry_fraction=%.2f mean_variance=%.1f over %d samples",
        video_path.name,
        report.blurry_fraction,
        report.mean_variance,
        len(samples),
    )
    return report
