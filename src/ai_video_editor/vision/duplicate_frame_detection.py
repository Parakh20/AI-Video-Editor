"""Perceptual-hash based duplicate / near-duplicate frame detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames

logger = get_logger(__name__)

HASH_SIZE = 8  # 8x8 -> 64-bit average hash
DEFAULT_HAMMING_THRESHOLD = 5  # bits; <= this many differing bits => near-duplicate


def average_hash(frame: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    """Compute a 64-bit average perceptual hash of a frame, as a bool array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    return small > mean


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    frame_indices: list[int]
    timestamps: list[float]


def find_duplicate_frames(
    video_path: Path,
    *,
    stride: int = 5,
    max_seconds: float | None = None,
    hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> list[DuplicateGroup]:
    """Find runs of consecutive sampled frames that are near-duplicates of each other."""
    groups: list[DuplicateGroup] = []
    current_group: DuplicateGroup | None = None
    prev_hash: np.ndarray | None = None
    prev_index: int | None = None
    prev_timestamp: float | None = None

    for index, timestamp, frame in iter_frames(video_path, stride=stride, max_seconds=max_seconds):
        frame_hash = average_hash(frame)

        if prev_hash is not None and hamming_distance(frame_hash, prev_hash) <= hamming_threshold:
            if current_group is None:
                current_group = DuplicateGroup(
                    frame_indices=[prev_index], timestamps=[prev_timestamp]  # type: ignore[list-item]
                )
                groups.append(current_group)
            current_group.frame_indices.append(index)
            current_group.timestamps.append(timestamp)
        else:
            current_group = None

        prev_hash = frame_hash
        prev_index = index
        prev_timestamp = timestamp

    logger.info(
        "duplicate_frames(%s): found %d near-duplicate runs", video_path.name, len(groups)
    )
    return groups
