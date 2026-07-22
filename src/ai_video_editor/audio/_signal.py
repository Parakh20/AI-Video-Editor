"""Shared low-level signal helpers used by multiple audio analysis modules."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


def load_audio(path: Path, *, sr: int | None = 22050, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio waveform from any ffmpeg-readable file (audio or video container).

    Returns (samples, sample_rate).
    """
    logger.info("Loading audio from %s (sr=%s)", path, sr)
    y, actual_sr = librosa.load(str(path), sr=sr, mono=mono)
    return y, int(actual_sr)


def rms_envelope(
    y: np.ndarray, sr: int, *, frame_length: int = 2048, hop_length: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms_values, frame_times_seconds) for an RMS energy envelope."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times


def rms_to_db(rms: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Convert an RMS envelope to dB relative to its own peak (0 dB = loudest frame).

    A track with essentially no signal at all (true digital silence, peak <= eps)
    has no meaningful "loudest frame" to be relative to; returning 0 dB for every
    frame in that case would make it look maximally loud rather than silent, so
    such tracks are reported as very quiet (-inf) everywhere instead.
    """
    if rms.size == 0:
        return rms
    peak = float(np.max(rms))
    if peak <= eps:
        return np.full_like(rms, -np.inf)
    return 20.0 * np.log10(np.maximum(rms, eps) / peak)


def merge_intervals(
    intervals: list[tuple[float, float]], *, max_gap: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent (start, end) intervals; gaps <= max_gap are bridged."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[tuple[float, float]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
