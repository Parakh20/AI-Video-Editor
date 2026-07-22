"""Silence interval detection via RMS energy thresholding."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ai_video_editor.audio._signal import load_audio, merge_intervals, rms_envelope, rms_to_db
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


class SilenceInterval(BaseModel):
    """A contiguous interval classified as silence."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silence(
    audio_path: Path,
    *,
    threshold_db: float = -40.0,
    min_duration: float = 0.3,
    sr: int = 22050,
) -> list[SilenceInterval]:
    """Detect intervals quieter than `threshold_db` (relative to the track's peak RMS).

    Intervals shorter than `min_duration` seconds are discarded as noise.
    """
    y, actual_sr = load_audio(audio_path, sr=sr)
    if y.size == 0:
        return []

    rms, times = rms_envelope(y, actual_sr)
    if rms.size == 0:
        return []

    db = rms_to_db(rms)
    is_quiet = db < threshold_db

    raw_intervals: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, quiet in enumerate(is_quiet):
        if quiet and start_idx is None:
            start_idx = i
        elif not quiet and start_idx is not None:
            raw_intervals.append((float(times[start_idx]), float(times[i])))
            start_idx = None
    if start_idx is not None:
        raw_intervals.append((float(times[start_idx]), float(times[-1])))

    merged = merge_intervals(raw_intervals)
    silences = [
        SilenceInterval(start=start, end=end)
        for start, end in merged
        if (end - start) >= min_duration
    ]

    logger.info("Detected %d silence interval(s) in %s", len(silences), audio_path)
    return silences
