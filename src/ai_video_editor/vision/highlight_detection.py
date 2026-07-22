"""Per-shot highlight scoring combining motion, sharpness, and scene signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.utils.video_io import iter_frames
from ai_video_editor.vision.motion_intensity import analyze_video_motion_intensity
from ai_video_editor.vision.scene_classifier import SceneCategory, classify_frame_scene
from ai_video_editor.vision.scene_detection import Shot, detect_shots
from ai_video_editor.vision.sharpness_scoring import score_clip_sharpness

logger = get_logger(__name__)

# Scene categories considered visually more engaging for a cycling POV edit;
# used as a mild boost, not a hard filter.
_ENGAGING_CATEGORIES = {SceneCategory.FOLIAGE, SceneCategory.SKY, SceneCategory.URBAN}

# Weights for the three signals feeding the highlight score.
_MOTION_WEIGHT = 0.45
_SHARPNESS_WEIGHT = 0.35
_SCENE_WEIGHT = 0.20


@dataclass(frozen=True, slots=True)
class ShotHighlight:
    shot: Shot
    motion_score: float
    sharpness_score: float
    scene_category: SceneCategory
    scene_bonus: float
    highlight_score: float  # 0-100


def _representative_scene(video_path: Path, shot: Shot) -> SceneCategory:
    mid = shot.start_seconds + shot.duration_seconds / 2
    for _, _, frame in iter_frames(video_path, stride=1, max_frames=1, start_seconds=mid):
        return classify_frame_scene(frame).category
    return SceneCategory.UNKNOWN


def score_shot_highlight(video_path: Path, shot: Shot) -> ShotHighlight:
    """Score a single shot's highlight-worthiness on a 0-100 scale."""
    duration = max(shot.duration_seconds, 0.1)

    motion = analyze_video_motion_intensity(
        video_path, start_seconds=shot.start_seconds, max_seconds=duration, stride=5
    )
    sharpness = score_clip_sharpness(
        video_path, start_seconds=shot.start_seconds, max_seconds=duration, stride=10
    )
    scene_category = _representative_scene(video_path, shot)
    scene_bonus = 100.0 if scene_category in _ENGAGING_CATEGORIES else 40.0

    highlight_score = (
        _MOTION_WEIGHT * motion.intensity_score
        + _SHARPNESS_WEIGHT * sharpness.mean_score
        + _SCENE_WEIGHT * scene_bonus
    )

    return ShotHighlight(
        shot=shot,
        motion_score=motion.intensity_score,
        sharpness_score=sharpness.mean_score,
        scene_category=scene_category,
        scene_bonus=scene_bonus,
        highlight_score=highlight_score,
    )


# Shots shorter than this are almost always scene-detector edge noise (a cut
# landing within a few frames of another) rather than real content, and are
# too short to sample optical flow from reliably.
MIN_SHOT_DURATION_SECONDS = 0.5


def detect_highlights(
    video_path: Path,
    *,
    scene_threshold: float = 27.0,
    max_seconds: float | None = None,
    min_shot_duration: float = MIN_SHOT_DURATION_SECONDS,
) -> list[ShotHighlight]:
    """Detect shots and score each for highlight-worthiness.

    Shots shorter than `min_shot_duration` are dropped before scoring (see
    MIN_SHOT_DURATION_SECONDS docstring above).
    """
    shots = detect_shots(video_path, threshold=scene_threshold, max_seconds=max_seconds)
    shots = [s for s in shots if s.duration_seconds >= min_shot_duration]
    if not shots:
        logger.warning("highlight_detection(%s): no shots met minimum duration", video_path.name)
        return []
    highlights = [score_shot_highlight(video_path, shot) for shot in shots]
    highlights.sort(key=lambda h: h.highlight_score, reverse=True)
    logger.info(
        "highlight_detection(%s): scored %d shots, top score=%.1f",
        video_path.name,
        len(highlights),
        highlights[0].highlight_score if highlights else 0.0,
    )
    return highlights
