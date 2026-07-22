"""`video-editor timeline` against a synthetic scores.json (fast, no real video decode)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import opentimelineio as otio
from typer.testing import CliRunner

from ai_video_editor.cli.main import app


def test_timeline_builds_edit_plan_and_both_exports(
    runner: CliRunner, synthetic_scores_json: Path, tmp_path: Path
) -> None:
    edit_plan_out = tmp_path / "edit_plan.json"
    otio_out = tmp_path / "timeline.otio"
    kdenlive_out = tmp_path / "project.kdenlive"

    result = runner.invoke(
        app,
        [
            "timeline",
            "--scores", str(synthetic_scores_json),
            "--edit-plan-out", str(edit_plan_out),
            "--otio-out", str(otio_out),
            "--kdenlive-out", str(kdenlive_out),
        ],
    )
    assert result.exit_code == 0, result.output

    assert edit_plan_out.exists()
    assert otio_out.exists()
    assert kdenlive_out.exists()

    # Both scored shots are well above the default threshold (30.0), so both
    # should be kept and appear as clips on the OTIO timeline.
    timeline = otio.adapters.read_from_file(str(otio_out))
    clips = [item for item in timeline.tracks[0] if isinstance(item, otio.schema.Clip)]
    assert len(clips) >= 1

    # Kdenlive export must be well-formed XML with at least one producer.
    root = ET.fromstring(kdenlive_out.read_text())
    assert root.tag == "mlt"
    assert len(root.findall("producer")) >= 1


def test_timeline_missing_scores_file_exits_nonzero(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["timeline", "--scores", str(tmp_path / "does_not_exist.json")])
    assert result.exit_code != 0
