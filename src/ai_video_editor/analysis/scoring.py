"""Weighted per-shot clip scoring engine (Phase 6).

Combines signals from :mod:`ai_video_editor.vision`, :mod:`ai_video_editor.audio`,
and :mod:`ai_video_editor.gps` into a single 0-100 "worth keeping" score per shot,
with every sub-score preserved for explainability.

Design decision — motion is not "more is better": cyclists want footage that
reads as *energetic but controlled*. A high-motion, high-shake segment (a
crash-cam blur) is worse than a high-motion, stable segment (a fast, smooth
descent), which in turn beats a low-motion, stable segment (a calm flat
stretch). This is modeled by multiplying the raw motion-intensity score by
the stability fraction (``stability_score / 100``) before it enters the
weighted sum — see :func:`_motion_quality`.

Design decision — GPS and music-timeline signals are best-effort: this
project's real footage has no GPS telemetry, so every GPS-derived sub-score
is ``None`` and excluded from the weighted sum (with weights renormalized
over whatever signals *are* available) whenever no track is supplied. The
same applies to a separate song track, which has no fixed time relationship
to a shot's timestamps; its output here is a *compatibility* score (does
this shot's energy suit a high-energy song section), not a timing
alignment — actual placement against the beat grid is the edit decision
engine's job (Phase 7).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ai_video_editor.audio.music_analysis import MusicAnalysis
from ai_video_editor.audio.silence_detection import SilenceInterval
from ai_video_editor.audio.speech_detection import SpeechSegment
from ai_video_editor.gps.models import EventType, RideEvent, RideMetrics
from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames
from ai_video_editor.vision.aesthetic_scoring import score_frame_aesthetics
from ai_video_editor.vision.blur_detection import DEFAULT_BLUR_THRESHOLD, laplacian_variance
from ai_video_editor.vision.brightness import frame_brightness
from ai_video_editor.vision.camera_shake import analyze_video_shake
from ai_video_editor.vision.motion_intensity import analyze_video_motion_intensity
from ai_video_editor.vision.scene_classifier import SceneCategory, classify_frame_scene
from ai_video_editor.vision.scene_detection import Shot, detect_shots
from ai_video_editor.vision.sharpness_scoring import score_clip_sharpness

logger = get_logger(__name__)

_ENGAGING_CATEGORIES = {SceneCategory.FOLIAGE, SceneCategory.SKY, SceneCategory.URBAN}

# Frames sampled per shot for the per-frame heuristics (aesthetic, brightness,
# scene). A handful of evenly-spaced samples is enough to be representative
# without re-decoding the whole shot multiple times.
_FRAME_SAMPLE_COUNT = 5


class ScoreWeights(BaseModel):
    """Relative weight of each signal in the overall 0-100 score.

    Weights are normalized over whichever signals are actually available for
    a given shot (see :func:`score_shot`), so they need not be re-tuned when
    GPS or a song track is absent — they simply drop out of the sum.
    """

    visual_quality: float = 0.20
    stability: float = 0.10
    motion: float = 0.20
    scenery: float = 0.10
    lighting: float = 0.10
    gps_speed: float = 0.05
    gps_events: float = 0.05
    audio_quality: float = 0.08
    speech_penalty: float = 0.02
    music_sync: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class SubScores(BaseModel):
    """Individual 0-100 signals feeding a shot's overall score, for explainability.

    Any field is ``None`` when its source data wasn't available for this shot
    (e.g. ``gps_speed`` when no GPS track was supplied).
    """

    visual_quality: float
    stability: float
    motion: float
    motion_raw: float
    scenery: float
    scene_category: str
    lighting: float
    gps_speed: float | None = None
    gps_events: float | None = None
    gps_event_details: list[str] = Field(default_factory=list)
    audio_quality: float | None = None
    speech_penalty: float | None = None
    music_sync: float | None = None


class SegmentScore(BaseModel):
    """A scored shot: source clip, time range, overall score, and sub-scores."""

    video_path: str
    shot_index: int
    start_seconds: float
    end_seconds: float
    overall_score: float
    sub_scores: SubScores
    gps_available: bool
    audio_available: bool
    music_available: bool

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _motion_quality(motion_raw: float, stability: float) -> float:
    """Motion is only rewarded to the extent it's stable — see module docstring."""
    return motion_raw * (stability / 100.0)


def _shot_blurry_fraction(video_path: Path, shot: Shot, *, stride: int = 10) -> float:
    """Fraction of sampled frames within this shot's own time range that are blurry."""
    variances = [
        laplacian_variance(frame)
        for _, _, frame in iter_frames(
            video_path, stride=stride, max_seconds=shot.duration_seconds, start_seconds=shot.start_seconds
        )
    ]
    if not variances:
        return 0.0
    return sum(v < DEFAULT_BLUR_THRESHOLD for v in variances) / len(variances)


def _sample_frame_metrics(video_path: Path, shot: Shot) -> tuple[float, float, SceneCategory]:
    """Average aesthetic score, average brightness, and midpoint scene category."""
    duration = shot.duration_seconds
    step = duration / (_FRAME_SAMPLE_COUNT + 1)
    aesthetic_scores: list[float] = []
    brightness_values: list[float] = []
    for i in range(1, _FRAME_SAMPLE_COUNT + 1):
        t = shot.start_seconds + step * i
        for _, _, frame in iter_frames(video_path, stride=1, max_frames=1, start_seconds=t):
            aesthetic_scores.append(score_frame_aesthetics(frame).score)
            brightness_values.append(frame_brightness(frame))

    mid = shot.start_seconds + duration / 2
    scene_category = SceneCategory.UNKNOWN
    for _, _, frame in iter_frames(video_path, stride=1, max_frames=1, start_seconds=mid):
        scene_category = classify_frame_scene(frame).category

    avg_aesthetic = sum(aesthetic_scores) / len(aesthetic_scores) if aesthetic_scores else 0.0
    if not brightness_values:
        lighting_score = 0.0
    else:
        mean_b = sum(brightness_values) / len(brightness_values)
        variance = sum((b - mean_b) ** 2 for b in brightness_values) / len(brightness_values)
        # Reward consistent lighting within the shot; a flickering exposure
        # (e.g. crossing in and out of shade) reads as lower quality.
        consistency = max(0.0, 1.0 - (variance**0.5) / (mean_b + 1e-6))
        lighting_score = 100.0 * consistency

    return avg_aesthetic, lighting_score, scene_category


def _gps_signals_for_shot(
    shot: Shot,
    gps_metrics: RideMetrics,
    gps_events: list[RideEvent],
    gps_offset_seconds: float,
) -> tuple[float | None, float | None, list[str]]:
    """Average speed score and event-boost score for the ride time range overlapping this shot."""
    window_start = shot.start_seconds + gps_offset_seconds
    window_end = shot.end_seconds + gps_offset_seconds

    samples_in_window = [
        s for s in gps_metrics.samples if window_start <= s.elapsed_s <= window_end
    ]
    if not samples_in_window:
        return None, None, []

    # Normalize speed to a 0-100 score; 50 km/h (~13.9 m/s) treated as a fast
    # descent/sprint ceiling for road cycling.
    speed_ceiling_mps = 50.0 / 3.6
    avg_speed_mps = sum(s.speed_mps for s in samples_in_window) / len(samples_in_window)
    speed_score = min(avg_speed_mps / speed_ceiling_mps, 1.0) * 100.0

    overlapping_events = [
        e for e in gps_events if e.start_s <= window_end and e.end_s >= window_start
    ]
    # Climbs, descents, and high-speed sections are the interesting moments
    # for a highlight edit; a stop is a mild penalty (nothing is happening).
    event_boost = 50.0  # neutral baseline
    details: list[str] = []
    for event in overlapping_events:
        details.append(f"{event.event_type.value}: {event.detail}")
        if event.event_type in (EventType.CLIMB, EventType.DESCENT, EventType.HIGH_SPEED):
            event_boost = max(event_boost, 90.0)
        elif event.event_type is EventType.INTERESTING_SEGMENT:
            event_boost = max(event_boost, 80.0)
        elif event.event_type is EventType.STOP:
            event_boost = min(event_boost, 20.0)

    return speed_score, event_boost, details


def _audio_signals_for_shot(
    shot: Shot,
    silence_intervals: list[SilenceInterval],
    speech_segments: list[SpeechSegment],
) -> tuple[float, float]:
    """Audio-quality (inverse of silence coverage) and speech-penalty scores."""
    duration = max(shot.duration_seconds, 1e-6)

    def _covered_seconds(intervals: list[tuple[float, float]]) -> float:
        covered = 0.0
        for start, end in intervals:
            overlap = min(end, shot.end_seconds) - max(start, shot.start_seconds)
            if overlap > 0:
                covered += overlap
        return covered

    silence_covered = _covered_seconds([(s.start, s.end) for s in silence_intervals])
    audio_quality = 100.0 * (1.0 - min(silence_covered / duration, 1.0))

    speech_covered = _covered_seconds([(s.start, s.end) for s in speech_segments])
    # Speech is a *minor* penalty (per spec) — a cycling POV edit generally
    # prefers ambient/music over narration, but it's not disqualifying.
    speech_fraction = min(speech_covered / duration, 1.0)
    speech_penalty = 100.0 * (1.0 - 0.3 * speech_fraction)

    return audio_quality, speech_penalty


def _music_sync_for_shot(motion_quality: float, music: MusicAnalysis) -> float:
    """Compatibility score: does this shot's energy suit *some* section of the song.

    A high-energy shot is scored by how close the song's peak energy gets to
    matching it (i.e. there exists a good home for it); see module docstring
    for why this isn't a timing alignment.
    """
    if not music.sections:
        return 50.0
    energy_rank = {"low": 0.2, "mid": 0.5, "high": 0.9}
    target = motion_quality / 100.0
    best = max(1.0 - abs(energy_rank[s.energy_label] - target) for s in music.sections)
    return best * 100.0


def score_shot(
    video_path: Path,
    shot: Shot,
    *,
    weights: ScoreWeights | None = None,
    gps_metrics: RideMetrics | None = None,
    gps_events: list[RideEvent] | None = None,
    gps_offset_seconds: float = 0.0,
    silence_intervals: list[SilenceInterval] | None = None,
    speech_segments: list[SpeechSegment] | None = None,
    music: MusicAnalysis | None = None,
) -> SegmentScore:
    """Score a single shot on a 0-100 scale, with full sub-score breakdown."""
    weights = weights or ScoreWeights()
    duration = max(shot.duration_seconds, 0.1)

    sharpness = score_clip_sharpness(
        video_path, start_seconds=shot.start_seconds, max_seconds=duration, stride=10
    )
    blurry_fraction = _shot_blurry_fraction(video_path, shot)
    shake = analyze_video_shake(
        video_path, start_seconds=shot.start_seconds, max_seconds=duration, stride=5
    )
    motion = analyze_video_motion_intensity(
        video_path, start_seconds=shot.start_seconds, max_seconds=duration, stride=5
    )
    aesthetic_score, lighting_score, scene_category = _sample_frame_metrics(video_path, shot)

    visual_quality = (
        0.4 * sharpness.mean_score + 0.4 * aesthetic_score + 0.2 * (1.0 - blurry_fraction) * 100.0
    )
    motion_quality = _motion_quality(motion.intensity_score, shake.stability_score)
    scenery_score = 90.0 if scene_category in _ENGAGING_CATEGORIES else 45.0

    gps_speed_score: float | None = None
    gps_event_score: float | None = None
    gps_details: list[str] = []
    if gps_metrics is not None and gps_metrics.samples:
        gps_speed_score, gps_event_score, gps_details = _gps_signals_for_shot(
            shot, gps_metrics, gps_events or [], gps_offset_seconds
        )
    gps_available = gps_speed_score is not None

    audio_available = silence_intervals is not None or speech_segments is not None
    audio_quality_score: float | None = None
    speech_penalty_score: float | None = None
    if audio_available:
        audio_quality_score, speech_penalty_score = _audio_signals_for_shot(
            shot, silence_intervals or [], speech_segments or []
        )

    music_available = music is not None
    music_sync_score: float | None = None
    if music is not None:
        music_sync_score = _music_sync_for_shot(motion_quality, music)

    # Renormalize weights over available signals so absent GPS/audio/music
    # doesn't silently deflate every score.
    available_weight_items: list[tuple[float, float]] = [
        (weights.visual_quality, visual_quality),
        (weights.stability, shake.stability_score),
        (weights.motion, motion_quality),
        (weights.scenery, scenery_score),
        (weights.lighting, lighting_score),
    ]
    if gps_speed_score is not None:
        available_weight_items.append((weights.gps_speed, gps_speed_score))
    if gps_event_score is not None:
        available_weight_items.append((weights.gps_events, gps_event_score))
    if audio_quality_score is not None:
        available_weight_items.append((weights.audio_quality, audio_quality_score))
    if speech_penalty_score is not None:
        available_weight_items.append((weights.speech_penalty, speech_penalty_score))
    if music_sync_score is not None:
        available_weight_items.append((weights.music_sync, music_sync_score))

    total_weight = sum(w for w, _ in available_weight_items) or 1.0
    overall = sum(w * v for w, v in available_weight_items) / total_weight

    sub_scores = SubScores(
        visual_quality=visual_quality,
        stability=shake.stability_score,
        motion=motion_quality,
        motion_raw=motion.intensity_score,
        scenery=scenery_score,
        scene_category=scene_category.value,
        lighting=lighting_score,
        gps_speed=gps_speed_score,
        gps_events=gps_event_score,
        gps_event_details=gps_details,
        audio_quality=audio_quality_score,
        speech_penalty=speech_penalty_score,
        music_sync=music_sync_score,
    )

    return SegmentScore(
        video_path=str(video_path),
        shot_index=shot.index,
        start_seconds=shot.start_seconds,
        end_seconds=shot.end_seconds,
        overall_score=float(overall),
        sub_scores=sub_scores,
        gps_available=gps_available,
        audio_available=audio_available,
        music_available=music_available,
    )


def score_video(
    video_path: Path,
    *,
    max_seconds: float | None = None,
    scene_threshold: float = 27.0,
    weights: ScoreWeights | None = None,
    gps_metrics: RideMetrics | None = None,
    gps_events: list[RideEvent] | None = None,
    gps_offset_seconds: float = 0.0,
    silence_intervals: list[SilenceInterval] | None = None,
    speech_segments: list[SpeechSegment] | None = None,
    music: MusicAnalysis | None = None,
    min_shot_duration: float = 0.5,
) -> list[SegmentScore]:
    """Detect shots in a video and score each one. See :func:`score_shot`."""
    shots = detect_shots(video_path, threshold=scene_threshold, max_seconds=max_seconds)
    shots = [s for s in shots if s.duration_seconds >= min_shot_duration]
    if not shots:
        logger.warning("score_video(%s): no shots met minimum duration", video_path.name)
        return []

    scores = [
        score_shot(
            video_path,
            shot,
            weights=weights,
            gps_metrics=gps_metrics,
            gps_events=gps_events,
            gps_offset_seconds=gps_offset_seconds,
            silence_intervals=silence_intervals,
            speech_segments=speech_segments,
            music=music,
        )
        for shot in shots
    ]
    logger.info(
        "score_video(%s): scored %d shots, mean=%.1f top=%.1f",
        video_path.name,
        len(scores),
        sum(s.overall_score for s in scores) / len(scores),
        max(s.overall_score for s in scores),
    )
    return scores


def write_scores_json(scores: list[SegmentScore], path: Path) -> None:
    """Serialize a list of scored segments to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(s.model_dump_json(indent=2) for s in scores) + "\n]"
    path.write_text(payload, encoding="utf-8")
    logger.info("Wrote %d scored segments to %s", len(scores), path)
