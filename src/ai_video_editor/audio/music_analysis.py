"""Background music analysis: energy envelope, spectral summary, section segmentation."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel, Field

from ai_video_editor.audio._signal import load_audio, rms_envelope
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_ENERGY_LABELS = ("low", "mid", "high")
_MIN_SECTION_DURATION = 4.0
_SMOOTHING_WINDOW = 21  # frames, odd for a centered moving average


class MusicSection(BaseModel):
    """A contiguous stretch of roughly-constant energy level."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    energy_label: str = Field(description="One of 'low', 'mid', 'high'.")

    @property
    def duration(self) -> float:
        return self.end - self.start


class MusicAnalysis(BaseModel):
    """Summary of a track's energy dynamics and coarse structure."""

    duration_seconds: float
    rms_times: list[float] = Field(description="Timestamps (s) for each rms_curve sample.")
    rms_curve: list[float] = Field(description="Normalized RMS energy, one value per time.")
    spectral_centroid_mean_hz: float
    spectral_bandwidth_mean_hz: float
    sections: list[MusicSection] = Field(
        description="Heuristic low/mid/high energy sections, useful as verse/chorus-style hints."
    )


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _segment_by_energy(
    smoothed_rms: np.ndarray, times: np.ndarray
) -> list[MusicSection]:
    if smoothed_rms.size == 0:
        return []

    low_q, high_q = np.quantile(smoothed_rms, [1 / 3, 2 / 3])

    def label_for(value: float) -> str:
        if value <= low_q:
            return "low"
        if value >= high_q:
            return "high"
        return "mid"

    labels = [label_for(v) for v in smoothed_rms]

    sections: list[MusicSection] = []
    seg_start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[seg_start_idx]:
            start_t = float(times[seg_start_idx])
            end_t = float(times[i - 1])
            sections.append(
                MusicSection(start=start_t, end=end_t, energy_label=labels[seg_start_idx])
            )
            seg_start_idx = i

    # Merge sections shorter than the minimum duration into a neighbor to avoid
    # a noisy, flickering section list.
    merged: list[MusicSection] = []
    for section in sections:
        if merged and section.duration < _MIN_SECTION_DURATION:
            prev = merged[-1]
            merged[-1] = MusicSection(
                start=prev.start, end=section.end, energy_label=prev.energy_label
            )
        else:
            merged.append(section)
    return merged


def analyze_music(audio_path: Path, *, sr: int = 22050) -> MusicAnalysis:
    """Compute an energy envelope, spectral summary, and heuristic section boundaries."""
    y, actual_sr = load_audio(audio_path, sr=sr)
    duration = float(librosa.get_duration(y=y, sr=actual_sr))

    if not np.any(y):
        logger.warning("Audio at %s is silent; returning empty music analysis", audio_path)
        return MusicAnalysis(
            duration_seconds=duration,
            rms_times=[],
            rms_curve=[],
            spectral_centroid_mean_hz=0.0,
            spectral_bandwidth_mean_hz=0.0,
            sections=[],
        )

    rms, times = rms_envelope(y, actual_sr)
    peak = float(np.max(rms)) if rms.size else 1.0
    normalized_rms = rms / max(peak, 1e-10)
    smoothed = _smooth(normalized_rms, _SMOOTHING_WINDOW)

    centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=actual_sr)[0]

    sections = _segment_by_energy(smoothed, times)

    logger.info(
        "Music analysis for %s: %.1fs, %d sections", audio_path, duration, len(sections)
    )
    return MusicAnalysis(
        duration_seconds=duration,
        rms_times=times.tolist(),
        rms_curve=normalized_rms.tolist(),
        spectral_centroid_mean_hz=float(np.mean(centroid)),
        spectral_bandwidth_mean_hz=float(np.mean(bandwidth)),
        sections=sections,
    )
