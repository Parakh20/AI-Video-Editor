"""Shared fixtures for timeline tests."""

from __future__ import annotations

from ai_video_editor.editing.decision_engine import EditAction


def make_action(
    video: str,
    start: float,
    end: float,
    *,
    kind: str = "keep",
    transition: str | None = "hard_cut",
    chapter: str | None = None,
    score: float = 50.0,
) -> EditAction:
    return EditAction(
        kind=kind,  # type: ignore[arg-type]
        video_path=video,
        start_seconds=start,
        end_seconds=end,
        source_shot_indices=[0],
        score=score,
        transition_in=transition,  # type: ignore[arg-type]
        beat_aligned=False,
        chapter_marker=chapter,
        explanation=f"test action for {video}",
    )
