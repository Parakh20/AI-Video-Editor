"""Shared fixtures: synthetic test videos for the vision module tests."""

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


@pytest.fixture
def static_gray_video(tmp_path: Path) -> Path:
    """30 identical mid-gray frames: static scene, no motion, no cuts."""
    frame = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    return _write_video(tmp_path / "static_gray.mp4", [frame.copy() for _ in range(30)])


@pytest.fixture
def bright_day_video(tmp_path: Path) -> Path:
    """30 bright, cool-toned frames simulating daylight."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = 230  # B
    frame[:, :, 1] = 220  # G
    frame[:, :, 2] = 200  # R
    return _write_video(tmp_path / "bright_day.mp4", [frame.copy() for _ in range(30)])


@pytest.fixture
def dark_night_video(tmp_path: Path) -> Path:
    """30 dark, warm-toned frames simulating night with artificial light."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = 10  # B
    frame[:, :, 1] = 20  # G
    frame[:, :, 2] = 45  # R (warm)
    return _write_video(tmp_path / "dark_night.mp4", [frame.copy() for _ in range(30)])


@pytest.fixture
def moving_gradient_video(tmp_path: Path) -> Path:
    """30 frames of a diagonal gradient that shifts each frame: strong, sharp motion."""
    frames = []
    base = np.tile(np.linspace(0, 255, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    for i in range(30):
        shifted = np.roll(base, shift=i * 3, axis=1)
        frame = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
        frames.append(frame)
    return _write_video(tmp_path / "moving_gradient.mp4", frames)


@pytest.fixture
def blurry_video(tmp_path: Path) -> Path:
    """30 frames of heavily gaussian-blurred noise: low Laplacian variance."""
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(30):
        noise = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(noise, (25, 25), 15)
        frames.append(blurred)
    return _write_video(tmp_path / "blurry.mp4", frames)


@pytest.fixture
def sharp_checkerboard_video(tmp_path: Path) -> Path:
    """30 frames of a sharp high-frequency checkerboard: high Laplacian variance."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    block = 8
    for y in range(0, HEIGHT, block):
        for x in range(0, WIDTH, block):
            if ((x // block) + (y // block)) % 2 == 0:
                frame[y : y + block, x : x + block] = 255
    return _write_video(tmp_path / "checkerboard.mp4", [frame.copy() for _ in range(30)])


@pytest.fixture
def scene_cut_video(tmp_path: Path) -> Path:
    """15 textured blue frames followed by 15 textured red frames: one hard cut.

    Each half has an internal gradient (not a flat color) so perceptual-hash
    based comparisons, which threshold against the frame's own mean, can
    actually distinguish the two halves.
    """
    h_gradient = np.tile(np.linspace(60, 200, WIDTH, dtype=np.uint8), (HEIGHT, 1))
    v_gradient = np.tile(np.linspace(60, 200, HEIGHT, dtype=np.uint8), (WIDTH, 1)).T
    blue = cv2.merge([h_gradient, np.zeros_like(h_gradient), np.zeros_like(h_gradient)])
    red = cv2.merge([np.zeros_like(v_gradient), np.zeros_like(v_gradient), v_gradient])
    frames = [blue.copy() for _ in range(15)] + [red.copy() for _ in range(15)]
    return _write_video(tmp_path / "scene_cut.mp4", frames)
