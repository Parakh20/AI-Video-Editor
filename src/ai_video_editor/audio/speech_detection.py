"""Heuristic speech-segment detection (energy + spectral shape based VAD).

This is not a trained voice-activity-detection model; it is a lightweight
signal-processing heuristic intended to flag likely speech regions (e.g. a
rider talking to camera) so downstream subtitle generation can be skipped
for silent/music-only stretches. For authoritative transcription, see
`subtitle_generation.py`, which runs Whisper directly.
"""

from __future__ import annotations

from pathlib import Path

import librosa
from pydantic import BaseModel, Field

from ai_video_editor.audio._signal import load_audio, merge_intervals, rms_envelope, rms_to_db
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

# Human voiced speech energy concentrates roughly in this band.
_SPEECH_CENTROID_MIN_HZ = 150.0
_SPEECH_CENTROID_MAX_HZ = 3500.0


class SpeechSegment(BaseModel):
    """A contiguous interval likely to contain speech."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_speech_segments(
    audio_path: Path,
    *,
    energy_threshold_db: float = -30.0,
    min_duration: float = 0.4,
    sr: int = 22050,
) -> list[SpeechSegment]:
    """Flag likely-speech intervals using an energy gate plus spectral-centroid gate.

    A frame is treated as speech-like when it is both loud enough (RMS above
    `energy_threshold_db` relative to the track's peak) and its spectral
    centroid falls within the typical vocal range, which helps reject
    steady-state music/wind noise that is loud but tonally different.
    """
    y, actual_sr = load_audio(audio_path, sr=sr)
    if y.size == 0:
        return []

    rms, times = rms_envelope(y, actual_sr)
    if rms.size == 0:
        return []
    db = rms_to_db(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=actual_sr)[0]
    # Align lengths defensively (rms/centroid can differ by a frame at edges).
    n = min(len(db), len(centroid), len(times))
    db, centroid, times = db[:n], centroid[:n], times[:n]

    is_speech_like = (db >= energy_threshold_db) & (
        (centroid >= _SPEECH_CENTROID_MIN_HZ) & (centroid <= _SPEECH_CENTROID_MAX_HZ)
    )

    raw_intervals: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, flag in enumerate(is_speech_like):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            raw_intervals.append((float(times[start_idx]), float(times[i])))
            start_idx = None
    if start_idx is not None:
        raw_intervals.append((float(times[start_idx]), float(times[-1])))

    merged = merge_intervals(raw_intervals, max_gap=0.2)
    segments = [
        SpeechSegment(start=start, end=end)
        for start, end in merged
        if (end - start) >= min_duration
    ]

    logger.info("Detected %d speech-like segment(s) in %s", len(segments), audio_path)
    return segments
