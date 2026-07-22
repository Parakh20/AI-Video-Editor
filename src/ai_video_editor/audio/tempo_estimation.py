"""Standalone tempo (BPM) estimation, independent of full beat-grid extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel, Field

from ai_video_editor.audio._signal import load_audio
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


class TempoEstimate(BaseModel):
    """Estimated tempo with a rough confidence proxy."""

    bpm: float = Field(description="Estimated tempo in beats per minute.")
    confidence: float = Field(
        description="0-1 proxy for confidence, derived from tempogram peak sharpness."
    )


def estimate_tempo(audio_path: Path, *, sr: int = 22050) -> TempoEstimate:
    """Estimate the dominant tempo of a track via a tempogram, without full beat tracking."""
    y, actual_sr = load_audio(audio_path, sr=sr)

    if not np.any(y):
        logger.warning("Audio at %s is silent; returning zero tempo", audio_path)
        return TempoEstimate(bpm=0.0, confidence=0.0)

    onset_env = librosa.onset.onset_strength(y=y, sr=actual_sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=actual_sr)
    tempo_frequencies = librosa.tempo_frequencies(tempogram.shape[0], sr=actual_sr)

    mean_tempogram = tempogram.mean(axis=1)
    # Ignore implausible tempo bins (very slow/fast) to stabilize the peak pick.
    valid = (tempo_frequencies >= 40.0) & (tempo_frequencies <= 220.0)
    mean_tempogram = np.where(valid, mean_tempogram, -np.inf)

    peak_idx = int(np.argmax(mean_tempogram))
    bpm = float(tempo_frequencies[peak_idx])

    finite_vals = mean_tempogram[np.isfinite(mean_tempogram)]
    if finite_vals.size > 1 and float(np.ptp(finite_vals)) > 0:
        peak_val = float(mean_tempogram[peak_idx])
        confidence = float((peak_val - finite_vals.min()) / np.ptp(finite_vals))
    else:
        confidence = 0.0

    logger.info("Estimated tempo %.1f BPM (confidence=%.2f) for %s", bpm, confidence, audio_path)
    return TempoEstimate(bpm=bpm, confidence=confidence)
