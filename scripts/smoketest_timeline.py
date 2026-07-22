"""Manual smoke test: build timeline.otio and project.kdenlive from the real edit plan.

Not part of the pytest suite (touches real output files by design). Run with:
    .venv/bin/python scripts/smoketest_timeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import opentimelineio as otio

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.timeline import (
    build_kdenlive_project,
    build_otio_timeline,
    write_kdenlive_project,
    write_otio,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    edit_plan_path = PROJECT_ROOT / "output" / "edit_plan_smoketest.json"
    actions_raw = json.loads(edit_plan_path.read_text(encoding="utf-8"))
    actions = [EditAction(**a) for a in actions_raw]
    print(f"Loaded {len(actions)} edit actions from {edit_plan_path}")

    timeline = build_otio_timeline(
        actions,
        name="Kolar Smoketest Timeline",
        song_path=str(PROJECT_ROOT / "assets" / "music" / "Kesha - TiK ToK (Lyrics) - 7clouds (youtube).mp3"),
        source_project="Kolar",
    )
    otio_path = PROJECT_ROOT / "output" / "timeline_smoketest.otio"
    write_otio(timeline, otio_path)
    read_back = otio.adapters.read_from_file(str(otio_path))
    clip_count = len([c for c in read_back.tracks[0] if isinstance(c, otio.schema.Clip)])
    print(f"OTIO: wrote and re-read {otio_path}, {clip_count} clips confirmed")

    kdenlive_path = PROJECT_ROOT / "output" / "project_smoketest.kdenlive"
    write_kdenlive_project(actions, kdenlive_path)
    from xml.etree.ElementTree import fromstring

    root = fromstring(kdenlive_path.read_text(encoding="utf-8"))
    producer_count = len(root.findall("producer"))
    print(f"Kdenlive: wrote and re-parsed {kdenlive_path}, {producer_count} producers confirmed")

    # Also exercise the raw string builder for completeness.
    xml_text = build_kdenlive_project(actions)
    assert len(xml_text) > 0


if __name__ == "__main__":
    main()
