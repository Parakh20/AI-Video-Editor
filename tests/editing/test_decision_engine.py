"""Tests for the automatic edit decision engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_editor.analysis.scoring import SegmentScore, SubScores
from ai_video_editor.audio.beat_detection import BeatGrid
from ai_video_editor.editing.decision_engine import build_edit_plan, write_edit_plan_json


def _sub_scores(**overrides: float | str | list[str]) -> SubScores:
    base: dict[str, object] = dict(
        visual_quality=70.0,
        stability=80.0,
        motion=60.0,
        motion_raw=70.0,
        scenery=90.0,
        scene_category="sky",
        lighting=80.0,
    )
    base.update(overrides)
    return SubScores(**base)  # type: ignore[arg-type]


def _segment(
    video: str, start: float, end: float, score: float, shot_index: int = 0
) -> SegmentScore:
    return SegmentScore(
        video_path=video,
        shot_index=shot_index,
        start_seconds=start,
        end_seconds=end,
        overall_score=score,
        sub_scores=_sub_scores(),
        gps_available=False,
        audio_available=False,
        music_available=False,
    )


def test_empty_input_returns_empty_plan() -> None:
    assert build_edit_plan([]) == []


def test_below_threshold_segment_is_dropped_with_explanation() -> None:
    scores = [_segment("clip.mp4", 0.0, 2.0, score=10.0)]
    plan = build_edit_plan(scores, threshold=30.0)
    assert len(plan) == 1
    assert plan[0].kind == "drop"
    assert "below threshold" in plan[0].explanation


def test_above_threshold_segment_is_kept_with_explanation() -> None:
    scores = [_segment("clip.mp4", 0.0, 2.0, score=75.0)]
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.0)
    assert len(plan) == 1
    assert plan[0].kind == "keep"
    assert plan[0].explanation.startswith("kept")
    assert plan[0].chapter_marker is not None


def test_adjacent_same_clip_highlights_are_merged() -> None:
    scores = [
        _segment("clip.mp4", 0.0, 2.0, score=80.0, shot_index=0),
        _segment("clip.mp4", 2.2, 4.0, score=85.0, shot_index=1),  # 0.2s gap
    ]
    plan = build_edit_plan(scores, threshold=30.0, merge_gap_tolerance=1.0, trim_seconds=0.0)
    assert len(plan) == 1
    action = plan[0]
    assert action.kind == "keep"
    assert action.start_seconds == 0.0
    assert action.end_seconds == 4.0
    assert action.source_shot_indices == [0, 1]
    assert "merged 2" in action.explanation


def test_different_source_clips_are_not_merged() -> None:
    scores = [
        _segment("clip_a.mp4", 0.0, 2.0, score=80.0),
        _segment("clip_b.mp4", 2.1, 4.0, score=85.0),
    ]
    plan = build_edit_plan(scores, threshold=30.0, merge_gap_tolerance=1.0)
    keeps = [a for a in plan if a.kind == "keep"]
    assert len(keeps) == 2


def test_gap_beyond_tolerance_is_not_merged() -> None:
    scores = [
        _segment("clip.mp4", 0.0, 2.0, score=80.0),
        _segment("clip.mp4", 5.0, 7.0, score=85.0),  # 3s gap, tolerance 1.0
    ]
    plan = build_edit_plan(scores, threshold=30.0, merge_gap_tolerance=1.0)
    keeps = [a for a in plan if a.kind == "keep"]
    assert len(keeps) == 2


def test_trim_shrinks_kept_segment_edges() -> None:
    scores = [_segment("clip.mp4", 0.0, 4.0, score=80.0)]
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.5)
    assert plan[0].start_seconds == pytest.approx(0.5)
    assert plan[0].end_seconds == pytest.approx(3.5)


def test_trim_does_not_invert_very_short_segments() -> None:
    scores = [_segment("clip.mp4", 0.0, 0.5, score=80.0)]
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=1.0)
    assert plan[0].start_seconds < plan[0].end_seconds


def test_beat_grid_aligns_cut_within_tolerance() -> None:
    scores = [_segment("clip.mp4", 1.0, 3.0, score=80.0)]
    beat_grid = BeatGrid(tempo_bpm=120.0, beat_times=[0.95, 1.9, 2.85], duration_seconds=10.0)
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.0, beat_tolerance=0.1, beat_grid=beat_grid)
    assert plan[0].beat_aligned is True
    assert plan[0].transition_in == "beat_cut"


def test_beat_grid_present_but_too_far_falls_back_to_hard_cut() -> None:
    scores = [_segment("clip.mp4", 1.0, 3.0, score=80.0)]
    beat_grid = BeatGrid(tempo_bpm=120.0, beat_times=[5.0, 6.0], duration_seconds=10.0)
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.0, beat_tolerance=0.1, beat_grid=beat_grid)
    assert plan[0].beat_aligned is False
    assert plan[0].transition_in == "hard_cut"


def test_no_beat_grid_merged_segment_uses_crossfade() -> None:
    scores = [
        _segment("clip.mp4", 0.0, 2.0, score=80.0, shot_index=0),
        _segment("clip.mp4", 2.1, 4.0, score=85.0, shot_index=1),
    ]
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.0)
    assert plan[0].transition_in == "crossfade"


def test_mixed_keep_and_drop_preserves_order_and_covers_all_input() -> None:
    scores = [
        _segment("clip.mp4", 0.0, 2.0, score=10.0),
        _segment("clip.mp4", 2.0, 4.0, score=80.0),
        _segment("clip.mp4", 4.0, 6.0, score=5.0),
    ]
    plan = build_edit_plan(scores, threshold=30.0, trim_seconds=0.0)
    kinds = [a.kind for a in plan]
    assert kinds == ["drop", "keep", "drop"]
    assert all(a.explanation for a in plan)


def test_write_edit_plan_json_roundtrip(tmp_path: Path) -> None:
    scores = [_segment("clip.mp4", 0.0, 2.0, score=80.0)]
    plan = build_edit_plan(scores, threshold=30.0)
    out_path = tmp_path / "edit_plan.json"
    write_edit_plan_json(plan, out_path)

    loaded = json.loads(out_path.read_text())
    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0]["kind"] == "keep"
    assert loaded[0]["explanation"]
