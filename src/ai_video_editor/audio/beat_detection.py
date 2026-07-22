"""Beat grid detection for music/audio tracks."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
from pydantic import BaseModel, Field

from ai_video_editor.audio._signal import load_audio
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


class BeatGrid(BaseModel):
    """Detected beat timestamps and overall tempo for a track."""

    tempo_bpm: float = Field(description="Estimated global tempo in beats per minute.")
    beat_times: list[float] = Field(description="Beat timestamps in seconds, ascending.")
    duration_seconds: float = Field(description="Total duration of the analyzed audio.")

    @property
    def beat_count(self) -> int:
        return len(self.beat_times)


def detect_beats(audio_path: Path, *, sr: int = 22050) -> BeatGrid:
    """Detect the beat grid and tempo of a track using onset-strength beat tracking.

    Works on any ffmpeg-readable audio/video file.
    """
    y, actual_sr = load_audio(audio_path, sr=sr)
    duration = float(librosa.get_duration(y=y, sr=actual_sr))

    if duration <= 0.0 or not np.any(y):
        logger.warning("Audio at %s is empty or silent; returning empty beat grid", audio_path)
        return BeatGrid(tempo_bpm=0.0, beat_times=[], duration_seconds=duration)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=actual_sr)
    tempo_val = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=actual_sr).tolist()

    logger.info(
        "Detected %d beats at %.1f BPM over %.1fs in %s",
        len(beat_times),
        tempo_val,
        duration,
        audio_path,
    )
    return BeatGrid(tempo_bpm=tempo_val, beat_times=beat_times, duration_seconds=duration)
