"""Real-data smoke test for `video-editor analyse`."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_video_editor.cli.main import app


@pytest.mark.slow
def test_analyse_writes_cache_for_smallest_real_clip(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        ["analyse", str(smallest_real_clip.parent), "--limit", "1", "--cache-dir", str(cache_dir)],
    )
    assert result.exit_code == 0, result.output

    cache_files = list(cache_dir.glob("*.analysis.json"))
    assert len(cache_files) == 1
    assert smallest_real_clip.stem in cache_files[0].name
