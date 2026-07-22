"""Shared fixtures for the analysis (scoring) module tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 10.0
WIDTH, HEIGHT = 160, 120


def _write_video(path: Path, frames: list[np.ndarray], fps: float = FPS) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _sharp_gradient_frame(offset: int) -> np.ndarray:
    """A high-frequency checkerboard-ish gradient frame, shifting with `offset`."""
    x = np.arange(WIDTH)
    y = np.arange(HEIGHT)
    xv, yv = np.meshgrid(x, y)
    pattern = ((xv + offset) // 4 + (yv // 4)) % 2 * 255
    frame = np.stack([pattern, pattern, pattern], axis=-1).astype(np.uint8)
    return frame


def _flat_dark_frame() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)


@pytest.fixture
def two_shot_video(tmp_path: Path) -> Path:
    """A ~4s video with a hard scene cut at 2s: dark/static first half, bright/moving+sharp second half.

    Built so PySceneDetect finds a boundary and the two halves score very
    differently (the second half should out-score the first on motion,
    sharpness, and brightness).
    """
    dark_frames = [_flat_dark_frame() for _ in range(int(FPS * 2))]
    moving_frames = [_sharp_gradient_frame(offset=i * 3) for i in range(int(FPS * 2))]
    return _write_video(tmp_path / "two_shot.mp4", dark_frames + moving_frames)
