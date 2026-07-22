"""Tests for OTIO timeline generation."""

from __future__ import annotations

from pathlib import Path

import opentimelineio as otio

from ai_video_editor.timeline.otio_builder import build_otio_timeline, write_otio
from tests.timeline.conftest import make_action


def test_empty_edit_plan_produces_empty_timeline() -> None:
    timeline = build_otio_timeline([])
    track = timeline.tracks[0]
    assert len(track) == 0


def test_dropped_actions_are_excluded() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0, kind="drop"),
        make_action("b.mp4", 0.0, 2.0, kind="keep"),
    ]
    timeline = build_otio_timeline(actions)
    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert len(clips) == 1
    assert clips[0].name.startswith("b_")


def test_clip_count_and_order_matches_kept_actions() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0),
        make_action("b.mp4", 5.0, 8.0),
        make_action("c.mp4", 1.0, 3.0),
    ]
    timeline = build_otio_timeline(actions)
    clips = [c for c in timeline.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert [c.name.split("_")[0] for c in clips] == ["a", "b", "c"]


def test_crossfade_inserts_transition() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0, transition="hard_cut"),
        make_action("a.mp4", 2.5, 4.0, transition="crossfade"),
    ]
    timeline = build_otio_timeline(actions)
    transitions = [t for t in timeline.tracks[0] if isinstance(t, otio.schema.Transition)]
    assert len(transitions) == 1


def test_chapter_marker_attached_to_clip() -> None:
    actions = [make_action("a.mp4", 0.0, 2.0, chapter="Sky — 0.0s")]
    timeline = build_otio_timeline(actions)
    clip = timeline.tracks[0][0]
    assert isinstance(clip, otio.schema.Clip)
    assert len(clip.markers) == 1
    assert clip.markers[0].name == "Sky — 0.0s"


def test_timeline_metadata_includes_generation_info() -> None:
    timeline = build_otio_timeline([], song_path="song.mp3", source_project="Kolar")
    meta = timeline.metadata["ai_video_editor"]
    assert meta["song_path"] == "song.mp3"
    assert meta["source_project"] == "Kolar"
    assert "generated_at" in meta


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0),
        make_action("b.mp4", 5.0, 8.0, transition="crossfade"),
    ]
    timeline = build_otio_timeline(actions, name="RoundTrip")
    out_path = tmp_path / "timeline.otio"
    write_otio(timeline, out_path)

    assert out_path.exists()
    read_back = otio.adapters.read_from_file(str(out_path))
    assert isinstance(read_back, otio.schema.Timeline)
    assert read_back.name == "RoundTrip"
    clips = [c for c in read_back.tracks[0] if isinstance(c, otio.schema.Clip)]
    assert len(clips) == 2
