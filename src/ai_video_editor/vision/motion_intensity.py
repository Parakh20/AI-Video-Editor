"""Overall motion intensity scoring from optical flow magnitude."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.optical_flow import FlowResult, analyze_video_flow

logger = get_logger(__name__)

# Mean flow magnitude (downscaled-pixel units) treated as "maximally intense"
# for normalization to a 0-100 scale.
MOTION_SATURATION_MAGNITUDE = 8.0


@dataclass(frozen=True, slots=True)
class MotionIntensityReport:
    intensity_score: float  # 0 (static) - 100 (very high motion)
    mean_magnitude: float
    sample_count: int


def intensity_score_from_flow(flow_results: list[FlowResult]) -> MotionIntensityReport:
    if not flow_results:
        raise ValueError("Need at least 1 flow result to estimate motion intensity")

    magnitudes = np.array([f.mean_magnitude for f in flow_results])
    mean_magnitude = float(np.mean(magnitudes))
    intensity_score = float(
        np.clip(mean_magnitude / MOTION_SATURATION_MAGNITUDE * 100.0, 0.0, 100.0)
    )
    return MotionIntensityReport(
        intensity_score=intensity_score, mean_magnitude=mean_magnitude, sample_count=len(flow_results)
    )


def analyze_video_motion_intensity(
    video_path: Path, *, stride: int = 5, max_seconds: float | None = None, start_seconds: float = 0.0
) -> MotionIntensityReport:
    """Estimate overall motion intensity (0-100) for a video clip or segment."""
    flow_results = analyze_video_flow(
        video_path, stride=stride, max_seconds=max_seconds, start_seconds=start_seconds
    )
    report = intensity_score_from_flow(flow_results)
    logger.info(
        "motion_intensity(%s): score=%.1f mean_magnitude=%.3f",
        video_path.name,
        report.intensity_score,
        report.mean_magnitude,
    )
    return report
