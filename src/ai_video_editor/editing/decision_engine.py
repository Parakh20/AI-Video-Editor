"""Automatic edit decision engine (Phase 7).

Consumes scored segments (:class:`ai_video_editor.analysis.scoring.SegmentScore`)
and an optional beat grid, and produces an ordered, fully-explained edit plan:
drop boring segments, merge adjacent highlights from the same source clip,
trim segment edges, choose a transition per cut, align cuts to the beat grid
when one is available and close enough, and generate chapter markers.

Every :class:`EditAction` carries a human-readable ``explanation`` — this is
a hard requirement, not a nice-to-have, so the plan is auditable without
re-deriving it from the raw scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_video_editor.analysis.scoring import SegmentScore
from ai_video_editor.audio.beat_detection import BeatGrid
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

ActionKind = Literal["keep", "drop"]
TransitionType = Literal["hard_cut", "beat_cut", "crossfade"]

DEFAULT_SCORE_THRESHOLD = 30.0
DEFAULT_MERGE_GAP_TOLERANCE = 1.0
DEFAULT_BEAT_TOLERANCE = 0.15
DEFAULT_TRIM_SECONDS = 0.15


class EditAction(BaseModel):
    """One decision in the edit plan: keep (possibly merged/trimmed) or drop a segment."""

    kind: ActionKind
    video_path: str
    start_seconds: float
    end_seconds: float
    source_shot_indices: list[int] = Field(
        description="Shot indices from the source SegmentScores this action was built from."
    )
    score: float | None = Field(default=None, description="Mean overall_score of merged shots.")
    transition_in: TransitionType | None = None
    beat_aligned: bool = False
    chapter_marker: str | None = None
    explanation: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _explain_drop(score: SegmentScore, threshold: float) -> str:
    return (
        f"dropped: score {score.overall_score:.1f} below threshold {threshold:.1f} "
        f"(motion={score.sub_scores.motion:.1f}, stability={score.sub_scores.stability:.1f}, "
        f"visual_quality={score.sub_scores.visual_quality:.1f})"
    )


def _explain_keep(scores: list[SegmentScore], merged: bool) -> str:
    mean_score = sum(s.overall_score for s in scores) / len(scores)
    if merged:
        return (
            f"kept (merged {len(scores)} adjacent highlights): mean score={mean_score:.1f}, "
            f"scene={scores[0].sub_scores.scene_category}"
        )
    s = scores[0]
    reasons = [f"score={s.overall_score:.1f}"]
    if s.sub_scores.motion > 60:
        reasons.append(f"energetic+stable (motion={s.sub_scores.motion:.1f})")
    if s.sub_scores.gps_events is not None and s.sub_scores.gps_events >= 80:
        reasons.append(f"GPS event: {'; '.join(s.sub_scores.gps_event_details)}")
    if s.sub_scores.scenery >= 90:
        reasons.append(f"engaging scene ({s.sub_scores.scene_category})")
    return f"kept: {', '.join(reasons)}"


def _nearest_beat(time: float, beat_times: list[float]) -> tuple[float, float] | None:
    """Return (beat_time, abs_offset) for the closest beat to `time`, or None if no beats."""
    if not beat_times:
        return None
    closest = min(beat_times, key=lambda b: abs(b - time))
    return closest, abs(closest - time)


def _trim(start: float, end: float, trim_seconds: float) -> tuple[float, float]:
    if end - start <= 2 * trim_seconds:
        return start, end
    return start + trim_seconds, end - trim_seconds


def _merge_kept_segments(
    kept: list[SegmentScore], *, merge_gap_tolerance: float
) -> list[list[SegmentScore]]:
    """Group consecutive kept segments from the same clip into merge-candidate groups."""
    groups: list[list[SegmentScore]] = []
    for score in kept:
        if groups:
            last_group = groups[-1]
            last = last_group[-1]
            same_clip = last.video_path == score.video_path
            gap = score.start_seconds - last.end_seconds
            if same_clip and 0 <= gap <= merge_gap_tolerance:
                last_group.append(score)
                continue
        groups.append([score])
    return groups


def build_edit_plan(
    scores: list[SegmentScore],
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    beat_grid: BeatGrid | None = None,
    merge_gap_tolerance: float = DEFAULT_MERGE_GAP_TOLERANCE,
    beat_tolerance: float = DEFAULT_BEAT_TOLERANCE,
    trim_seconds: float = DEFAULT_TRIM_SECONDS,
) -> list[EditAction]:
    """Build a fully-explained, ordered edit plan from scored segments.

    Segments are processed in the order given; segments from the same source
    clip that are both kept and close together (within `merge_gap_tolerance`
    seconds) are merged into a single action. Dropped segments are also
    included in the output (as `kind="drop"` actions) so the plan is a
    complete, auditable account of every input segment.
    """
    if not scores:
        return []

    actions: list[EditAction] = []
    kept_run: list[SegmentScore] = []

    def flush_kept_run() -> None:
        if not kept_run:
            return
        for group in _merge_kept_segments(kept_run, merge_gap_tolerance=merge_gap_tolerance):
            merged = len(group) > 1
            start = group[0].start_seconds
            end = group[-1].end_seconds
            trimmed_start, trimmed_end = _trim(start, end, trim_seconds)

            transition: TransitionType = "hard_cut"
            beat_aligned = False
            if beat_grid is not None and beat_grid.beat_times:
                nearest = _nearest_beat(trimmed_start, beat_grid.beat_times)
                if nearest is not None and nearest[1] <= beat_tolerance:
                    transition = "beat_cut"
                    beat_aligned = True
            elif merged:
                transition = "crossfade"

            actions.append(
                EditAction(
                    kind="keep",
                    video_path=group[0].video_path,
                    start_seconds=trimmed_start,
                    end_seconds=trimmed_end,
                    source_shot_indices=[s.shot_index for s in group],
                    score=sum(s.overall_score for s in group) / len(group),
                    transition_in=transition,
                    beat_aligned=beat_aligned,
                    chapter_marker=f"{group[0].sub_scores.scene_category.title()} — {trimmed_start:.1f}s",
                    explanation=_explain_keep(group, merged),
                )
            )
        kept_run.clear()

    for score in scores:
        if score.overall_score < threshold:
            flush_kept_run()
            actions.append(
                EditAction(
                    kind="drop",
                    video_path=score.video_path,
                    start_seconds=score.start_seconds,
                    end_seconds=score.end_seconds,
                    source_shot_indices=[score.shot_index],
                    score=score.overall_score,
                    explanation=_explain_drop(score, threshold),
                )
            )
        else:
            kept_run.append(score)
    flush_kept_run()

    kept_count = sum(1 for a in actions if a.kind == "keep")
    logger.info(
        "build_edit_plan: %d actions (%d keep, %d drop) from %d input segments",
        len(actions),
        kept_count,
        len(actions) - kept_count,
        len(scores),
    )
    return actions


def write_edit_plan_json(actions: list[EditAction], path: Path) -> None:
    """Serialize an edit plan to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(a.model_dump_json(indent=2) for a in actions) + "\n]"
    path.write_text(payload, encoding="utf-8")
    logger.info("Wrote %d edit actions to %s", len(actions), path)
