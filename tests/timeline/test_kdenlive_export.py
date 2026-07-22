"""Tests for Kdenlive (MLT XML) project generation."""

from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import fromstring

from ai_video_editor.timeline.kdenlive_export import (
    build_kdenlive_project,
    write_kdenlive_project,
)
from tests.timeline.conftest import make_action


def test_empty_edit_plan_produces_valid_empty_project() -> None:
    xml_text = build_kdenlive_project([])
    root = fromstring(xml_text)
    assert root.tag == "mlt"
    assert root.find("playlist") is not None
    assert len(root.findall("producer")) == 0


def test_dropped_actions_produce_no_producers() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0, kind="drop"),
        make_action("b.mp4", 0.0, 2.0, kind="keep"),
    ]
    xml_text = build_kdenlive_project(actions)
    root = fromstring(xml_text)
    producers = root.findall("producer")
    assert len(producers) == 1


def test_producer_count_matches_kept_actions() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0),
        make_action("b.mp4", 5.0, 8.0),
        make_action("c.mp4", 1.0, 3.0),
    ]
    xml_text = build_kdenlive_project(actions)
    root = fromstring(xml_text)
    assert len(root.findall("producer")) == 3
    playlist = root.find("playlist")
    assert playlist is not None
    assert len(playlist.findall("entry")) == 3


def test_resource_path_is_preserved_in_producer_property() -> None:
    actions = [make_action("/tmp/some clip.mp4", 0.0, 2.0)]
    xml_text = build_kdenlive_project(actions)
    root = fromstring(xml_text)
    producer = root.find("producer")
    assert producer is not None
    resource_prop = next(
        p for p in producer.findall("property") if p.get("name") == "resource"
    )
    assert resource_prop.text == "/tmp/some clip.mp4"


def test_crossfade_produces_transition_element() -> None:
    actions = [
        make_action("a.mp4", 0.0, 2.0, transition="hard_cut"),
        make_action("a.mp4", 2.5, 4.0, transition="crossfade"),
    ]
    xml_text = build_kdenlive_project(actions)
    root = fromstring(xml_text)
    assert len(root.findall("transition")) == 1


def test_tractor_and_track_present() -> None:
    actions = [make_action("a.mp4", 0.0, 2.0)]
    xml_text = build_kdenlive_project(actions)
    root = fromstring(xml_text)
    tractor = root.find("tractor")
    assert tractor is not None
    assert tractor.find("track") is not None
    assert tractor.find("track").get("producer") == "playlist0"


def test_write_project_file(tmp_path: Path) -> None:
    actions = [make_action("a.mp4", 0.0, 2.0)]
    out_path = tmp_path / "project.kdenlive"
    write_kdenlive_project(actions, out_path)

    assert out_path.exists()
    root = fromstring(out_path.read_text(encoding="utf-8"))
    assert root.tag == "mlt"
