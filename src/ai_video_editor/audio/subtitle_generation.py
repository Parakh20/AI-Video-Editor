"""Subtitle/transcript generation via OpenAI Whisper."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import whisper
from pydantic import BaseModel, Field

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_SIZE = "base"
_loaded_models: dict[str, "whisper.Whisper"] = {}


class TranscriptSegment(BaseModel):
    """A single timed transcript segment."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")
    text: str = Field(description="Transcribed text for this segment.")


class SubtitleResult(BaseModel):
    """Full transcription result for one audio/video source."""

    segments: list[TranscriptSegment]
    srt_text: str = Field(description="The transcript rendered as SRT-formatted subtitles.")
    language: str | None = Field(default=None, description="Detected/forced language code.")

    @property
    def has_speech(self) -> bool:
        return len(self.segments) > 0


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model(model_size: str) -> "whisper.Whisper":
    if model_size not in _loaded_models:
        device = _select_device()
        logger.info("Loading Whisper model '%s' on %s", model_size, device)
        _loaded_models[model_size] = whisper.load_model(model_size, device=device)
    return _loaded_models[model_size]


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_srt(segments: list[TranscriptSegment]) -> str:
    if not segments:
        return ""
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _has_audio_stream(path: Path) -> bool:
    """Probe via ffprobe whether ``path`` has at least one audio stream.

    Some real action-cam clips (confirmed in this project's test footage)
    have no audio track at all; Whisper's internal ffmpeg decode step raises
    a hard error in that case rather than returning silence, so callers must
    check first.
    """
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        logger.warning("ffprobe failed for %s (%s); assuming no audio stream", path, probe.stderr.strip())
        return False
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    return len(streams) > 0


def generate_subtitles(
    audio_path: Path,
    *,
    model_size: str = _DEFAULT_MODEL_SIZE,
    language: str | None = None,
) -> SubtitleResult:
    """Transcribe an audio or video file into timed subtitle segments + SRT text.

    Accepts any ffmpeg-readable source (Whisper decodes audio via ffmpeg
    internally, so passing a video file directly is fine). Returns an empty
    result (not an error) when no speech is detected, or when the source has
    no audio stream at all.
    """
    if not _has_audio_stream(audio_path):
        logger.info("Skipping transcription of %s: no audio stream", audio_path)
        return SubtitleResult(segments=[], srt_text="", language=None)

    model = _get_model(model_size)
    logger.info("Transcribing %s (model=%s)", audio_path, model_size)

    result = model.transcribe(str(audio_path), language=language, verbose=False)

    raw_segments = result.get("segments", [])
    segments = [
        TranscriptSegment(start=float(seg["start"]), end=float(seg["end"]), text=str(seg["text"]))
        for seg in raw_segments
        if str(seg.get("text", "")).strip()
    ]

    detected_language = result.get("language")
    srt_text = _build_srt(segments)

    logger.info(
        "Transcription of %s produced %d segment(s), language=%s",
        audio_path,
        len(segments),
        detected_language,
    )
    return SubtitleResult(segments=segments, srt_text=srt_text, language=detected_language)
