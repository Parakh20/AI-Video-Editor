"""Camera shake / stability estimation from optical flow.

Shake is distinguished from intentional motion (panning) by looking at the
frame-to-frame *variance* of the global flow direction: a smooth pan has a
consistent (dx, dy) across frames, while handheld shake has a high-variance,
jittery direction even at similar magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.optical_flow import FlowResult, analyze_video_flow

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ShakeReport:
    stability_score: float  # 0 (very shaky) - 100 (very stable)
    direction_variance: float
    mean_magnitude: float
    sample_count: int


def stability_score_from_flow(flow_results: list[FlowResult]) -> ShakeReport:
    """Derive a stability score from a sequence of per-frame flow results."""
    if len(flow_results) < 2:
        raise ValueError("Need at least 2 flow results to estimate shake")

    dx = np.array([f.mean_dx for f in flow_results])
    dy = np.array([f.mean_dy for f in flow_results])
    magnitude = np.array([f.mean_magnitude for f in flow_results])

    direction_variance = float(np.var(dx) + np.var(dy))
    # Empirically, direction_variance above ~4.0 (in downscaled-pixel units,
    # see optical_flow.analyze_video_flow) reads as visibly shaky footage.
    stability_score = float(np.clip(100.0 - (direction_variance / 4.0) * 100.0, 0.0, 100.0))

    return ShakeReport(
        stability_score=stability_score,
        direction_variance=direction_variance,
        mean_magnitude=float(np.mean(magnitude)),
        sample_count=len(flow_results),
    )


def analyze_video_shake(
    video_path: Path, *, stride: int = 5, max_seconds: float | None = None, start_seconds: float = 0.0
) -> ShakeReport:
    """Estimate camera shake/stability for a video clip."""
    flow_results = analyze_video_flow(
        video_path, stride=stride, max_seconds=max_seconds, start_seconds=start_seconds
    )
    report = stability_score_from_flow(flow_results)
    logger.info(
        "shake(%s): stability=%.1f direction_variance=%.3f",
        video_path.name,
        report.stability_score,
        report.direction_variance,
    )
    return report
