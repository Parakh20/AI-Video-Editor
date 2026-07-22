"""`video-editor render --dry-run` against a synthetic edit plan (no real ffmpeg encode)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ai_video_editor.cli.main import app


def _write_edit_plan(tmp_path: Path) -> Path:
    fake_video = tmp_path / "fake_clip.mp4"
    fake_video.touch()
    actions = [
        {
            "kind": "keep",
            "video_path": str(fake_video),
            "start_seconds": 0.0,
            "end_seconds": 3.0,
            "source_shot_indices": [0],
            "score": 82.0,
            "transition_in": "hard_cut",
            "beat_aligned": False,
            "chapter_marker": "Sky — 0.0s",
            "explanation": "kept: score=82.0",
        }
    ]
    path = tmp_path / "edit_plan.json"
    path.write_text(json.dumps(actions, indent=2), encoding="utf-8")
    return path


def test_render_dry_run_does_not_invoke_ffmpeg(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    output = tmp_path / "out.mp4"

    result = runner.invoke(
        app,
        ["render", "--edit-plan", str(edit_plan), "--preset", "1080p", "--output", str(output), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert not output.exists()
    assert "dry run" in result.output.lower()


def test_render_unknown_preset_fails(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    result = runner.invoke(
        app,
        ["render", "--edit-plan", str(edit_plan), "--preset", "not_a_real_preset", "--dry-run"],
    )
    assert result.exit_code != 0


def test_render_missing_edit_plan_exits_nonzero(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["render", "--edit-plan", str(tmp_path / "missing.json")])
    assert result.exit_code != 0


FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_ride.gpx"


def test_render_dry_run_reports_gps_overlay(runner: CliRunner, tmp_path: Path) -> None:
    edit_plan = _write_edit_plan(tmp_path)
    result = runner.invoke(
        app,
        [
            "render",
            "--edit-plan", str(edit_plan),
            "--preset", "1080p",
            "--gps", str(FIXTURE_GPX),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "with gps overlay" in result.output.lower()


def test_gps_overlay_fragment_is_time_synced(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_overlay_fragment, load_edit_plan_json

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    overlay = _gps_overlay_fragment(FIXTURE_GPX, kept, crossfade_s=0.5)
    assert overlay is not None
    # Time-synced: multiple drawtext filters, each gated on an output-time window.
    assert overlay.count("drawtext=text=") >= 2
    assert "enable='between(t," in overlay


def test_gps_overlay_fragment_none_without_gps(tmp_path: Path) -> None:
    from ai_video_editor.cli.commands.render import _gps_overlay_fragment, load_edit_plan_json

    kept = load_edit_plan_json(_write_edit_plan(tmp_path))
    assert _gps_overlay_fragment(None, kept, crossfade_s=0.5) is None
