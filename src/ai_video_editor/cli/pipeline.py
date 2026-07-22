"""Shared helpers used by multiple CLI commands: clip discovery, per-clip
analysis, and GPS loading. Kept out of the command modules so `analyse`,
`score`, and `full` don't duplicate this logic.
"""

from __future__ import annotations

from pathlib import Path

from ai_video_editor.analysis.scoring import ScoreWeights, SegmentScore, score_video
from ai_video_editor.audio.music_analysis import MusicAnalysis, analyze_music
from ai_video_editor.audio.silence_detection import SilenceInterval, detect_silence
from ai_video_editor.audio.speech_detection import SpeechSegment, detect_speech_segments
from ai_video_editor.gps.event_detection import detect_events
from ai_video_editor.gps.metrics import compute_metrics
from ai_video_editor.gps.models import RideEvent, RideMetrics
from ai_video_editor.gps.parsers import load_track
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".MP4", ".MOV", ".mkv", ".avi"}


def discover_clips(input_dir: Path, *, limit: int | None = None) -> list[Path]:
    """Find video files under `input_dir`, sorted smallest-first.

    Smallest-first is deliberate: it makes `--limit N` a cheap, predictable
    way to smoke-test the pipeline against real footage without waiting on
    this project's largest clips (several GB / many minutes each).
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    clips = sorted(
        (p for p in input_dir.iterdir() if p.suffix in VIDEO_EXTENSIONS and p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    if limit is not None:
        clips = clips[:limit]
    logger.info(
        "discover_clips(%s): found %d clip(s)%s",
        input_dir,
        len(clips),
        f" (limited to {limit})" if limit is not None else "",
    )
    for clip in clips:
        size_mb = clip.stat().st_size / (1024 * 1024)
        logger.info("  - %s (%.1f MB)", clip.name, size_mb)
    return clips


def load_gps(gps_path: Path | None) -> tuple[RideMetrics | None, list[RideEvent]]:
    """Load and derive GPS metrics + events, or (None, []) if no path given.

    Every downstream caller must handle the None case -- there is
    deliberately no synthetic fallback here, since a fabricated GPS track
    would silently misrepresent a real ride.
    """
    if gps_path is None:
        return None, []
    track = load_track(gps_path)
    metrics = compute_metrics(track)
    events = detect_events(metrics)
    logger.info(
        "load_gps(%s): %d samples, %d events", gps_path, len(metrics.samples), len(events)
    )
    return metrics, events


def load_music(music_path: Path | None) -> MusicAnalysis | None:
    if music_path is None:
        return None
    return analyze_music(music_path)


def analyse_clip_audio(
    clip_path: Path,
) -> tuple[list[SilenceInterval], list[SpeechSegment]]:
    """Best-effort silence/speech analysis for one clip's audio track.

    Some real source clips have no audio stream at all (confirmed on this
    project's own footage); librosa/soundfile decoding failures are logged
    and treated as "no audio available" rather than aborting the clip.
    """
    try:
        silence = detect_silence(clip_path)
        speech = detect_speech_segments(clip_path)
        return silence, speech
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any decode failure means "no audio"
        logger.warning("analyse_clip_audio(%s): audio analysis unavailable (%s)", clip_path.name, exc)
        return [], []


def score_clip(
    clip_path: Path,
    *,
    weights: ScoreWeights,
    scene_threshold: float,
    min_shot_duration: float,
    max_seconds: float | None,
    gps_metrics: RideMetrics | None,
    gps_events: list[RideEvent],
    music: MusicAnalysis | None,
    use_audio: bool = True,
) -> list[SegmentScore]:
    """Score every shot in one clip, wiring in whatever optional signals are available."""
    silence, speech = analyse_clip_audio(clip_path) if use_audio else ([], [])
    return score_video(
        clip_path,
        max_seconds=max_seconds,
        scene_threshold=scene_threshold,
        weights=weights,
        gps_metrics=gps_metrics,
        gps_events=gps_events,
        silence_intervals=silence,
        speech_segments=speech,
        music=music,
        min_shot_duration=min_shot_duration,
    )
