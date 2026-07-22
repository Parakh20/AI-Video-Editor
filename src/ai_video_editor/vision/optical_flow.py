"""Dense optical flow computation between consecutive frames (Farneback)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FlowResult:
    frame_index: int
    timestamp: float
    mean_magnitude: float
    mean_dx: float
    mean_dy: float


def compute_flow(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Farneback dense optical flow between two grayscale frames. Returns HxWx2 (dx, dy)."""
    initial_flow = np.zeros((*prev_gray.shape[:2], 2), dtype=np.float32)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        initial_flow,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def analyze_video_flow(
    video_path: Path,
    *,
    stride: int = 5,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
    resize_width: int = 320,
) -> list[FlowResult]:
    """Compute per-sampled-frame-pair optical flow statistics across a video.

    Frames are downscaled to `resize_width` before flow computation for speed;
    the reported magnitudes are in downscaled-pixel units, which is fine for
    relative comparisons (shake/motion-intensity scoring) but not absolute
    real-world motion.
    """
    results: list[FlowResult] = []
    prev_gray: np.ndarray | None = None

    for index, timestamp, frame in iter_frames(
        video_path, stride=stride, max_seconds=max_seconds, start_seconds=start_seconds
    ):
        h, w = frame.shape[:2]
        scale = resize_width / w
        small = cv2.resize(frame, (resize_width, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = compute_flow(prev_gray, gray)
            magnitude = np.linalg.norm(flow, axis=2)
            results.append(
                FlowResult(
                    frame_index=index,
                    timestamp=timestamp,
                    mean_magnitude=float(np.mean(magnitude)),
                    mean_dx=float(np.mean(flow[..., 0])),
                    mean_dy=float(np.mean(flow[..., 1])),
                )
            )
        prev_gray = gray

    logger.info("optical_flow(%s): computed %d flow pairs", video_path.name, len(results))
    return results


def iter_video_flow(
    video_path: Path, *, stride: int = 5, max_seconds: float | None = None, resize_width: int = 320
) -> Iterator[FlowResult]:
    """Streaming variant of analyze_video_flow for large videos."""
    prev_gray: np.ndarray | None = None
    for index, timestamp, frame in iter_frames(video_path, stride=stride, max_seconds=max_seconds):
        h, w = frame.shape[:2]
        scale = resize_width / w
        small = cv2.resize(frame, (resize_width, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = compute_flow(prev_gray, gray)
            magnitude = np.linalg.norm(flow, axis=2)
            yield FlowResult(
                frame_index=index,
                timestamp=timestamp,
                mean_magnitude=float(np.mean(magnitude)),
                mean_dx=float(np.mean(flow[..., 0])),
                mean_dy=float(np.mean(flow[..., 1])),
            )
        prev_gray = gray
