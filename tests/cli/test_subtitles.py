"""Real-data smoke test for `video-editor subtitles`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_video_editor.cli.main import app


@pytest.mark.slow
def test_subtitles_writes_srt_for_smallest_real_clip(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "subs"
    result = runner.invoke(
        app,
        [
            "subtitles",
            str(smallest_real_clip),
            "--model-size", "tiny",
            "--output-dir", str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    srt_path = output_dir / f"{smallest_real_clip.stem}.srt"
    assert srt_path.exists()
    assert "Transcribed 1 clip(s)" in result.output


@pytest.mark.slow
def test_subtitles_on_directory_respects_limit(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "subs_dir"
    result = runner.invoke(
        app,
        [
            "subtitles",
            str(smallest_real_clip.parent),
            "--model-size", "tiny",
            "--output-dir", str(output_dir),
            "--limit", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(output_dir.glob("*.srt"))) == 1
