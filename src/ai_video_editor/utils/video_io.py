"""Shared OpenCV video I/O helpers used across the vision analysis modules."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Basic technical metadata for a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def get_video_info(video_path: Path) -> VideoInfo:
    """Probe a video file's basic technical metadata via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video file: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return VideoInfo(path=video_path, width=width, height=height, fps=fps, frame_count=frame_count)


def iter_frames(
    video_path: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame_bgr) from a video file.

    `stride` skips frames (stride=5 samples every 5th frame). `max_frames` and
    `max_seconds` bound how much of the video is read (max_seconds is relative
    to `start_seconds`, i.e. a segment duration), whichever hits first.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    start_frame = int(start_seconds * fps)
    frame_budget = start_frame + int(max_seconds * fps) if max_seconds is not None else None

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    yielded = 0
    index = start_frame
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_budget is not None and index >= frame_budget:
                break
            if index % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                timestamp = index / fps
                yield index, timestamp, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        cap.release()
    logger.debug("iter_frames(%s): yielded %d frames (stride=%d)", video_path.name, yielded, stride)
