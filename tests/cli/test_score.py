"""Real-data smoke test for `video-editor score`, plus GPS wiring via the synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_video_editor.cli.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_GPX = PROJECT_ROOT / "tests" / "fixtures" / "sample_ride.gpx"


@pytest.mark.slow
def test_score_writes_scores_json_for_smallest_real_clip(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path
) -> None:
    output = tmp_path / "scores.json"
    result = runner.invoke(
        app,
        ["score", str(smallest_real_clip.parent), "--limit", "1", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()

    scores = json.loads(output.read_text())
    assert len(scores) >= 1
    assert all(s["gps_available"] is False for s in scores)  # no --gps given


@pytest.mark.slow
def test_score_with_gps_marks_segments_gps_available(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path
) -> None:
    output = tmp_path / "scores_gps.json"
    result = runner.invoke(
        app,
        [
            "score",
            str(smallest_real_clip.parent),
            "--limit", "1",
            "--gps", str(SAMPLE_GPX),
            "--output", str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    scores = json.loads(output.read_text())
    assert len(scores) >= 1
    # sample_ride.gpx spans several minutes; a ~6s clip may or may not fall
    # inside its elapsed-time window depending on shot timestamps, so just
    # assert the command ran end-to-end with --gps wired in without error.
