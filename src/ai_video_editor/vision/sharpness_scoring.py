"""Per-clip sharpness scoring, normalized to a 0-100 scale for use in clip scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.blur_detection import laplacian_variance
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)

# Laplacian variance value treated as "maximally sharp" for normalization.
# Frames at or above this map to a sharpness score of 100.
SHARPNESS_SATURATION_VARIANCE = 800.0


def normalize_sharpness(variance: float) -> float:
    """Map a raw Laplacian variance to a 0-100 sharpness score."""
    return float(np.clip(variance / SHARPNESS_SATURATION_VARIANCE * 100.0, 0.0, 100.0))


@dataclass(frozen=True, slots=True)
class SharpnessReport:
    mean_score: float
    p25_score: float
    p75_score: float
    sample_count: int


def score_clip_sharpness(
    video_path: Path,
    *,
    stride: int = 15,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> SharpnessReport:
    """Compute an aggregate sharpness score (0-100) for a clip."""
    scores = [
        normalize_sharpness(laplacian_variance(frame))
        for _, _, frame in iter_frames(
            video_path, stride=stride, max_seconds=max_seconds, start_seconds=start_seconds
        )
    ]
    if not scores:
        raise ValueError(f"No frames read from {video_path}")

    report = SharpnessReport(
        mean_score=float(np.mean(scores)),
        p25_score=float(np.percentile(scores, 25)),
        p75_score=float(np.percentile(scores, 75)),
        sample_count=len(scores),
    )
    logger.info(
        "sharpness(%s): mean=%.1f p25=%.1f p75=%.1f over %d samples",
        video_path.name,
        report.mean_score,
        report.p25_score,
        report.p75_score,
        report.sample_count,
    )
    return report
