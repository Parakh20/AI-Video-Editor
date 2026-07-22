"""Real end-to-end smoke test for `video-editor full` (--dry-run, --limit 1 for speed).

The genuine, non-dry-run, multi-clip end-to-end run (the project's actual
success criterion) is exercised separately as a documented manual smoke
test -- see docs/pipeline_smoke_test.md -- since it legitimately takes
several minutes against real multi-gigabyte footage and doesn't belong in
the default test suite's runtime budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_video_editor.cli.main import app


@pytest.mark.slow
def test_full_dry_run_single_clip_produces_all_artifacts(
    runner: CliRunner, smallest_real_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `full` writes to a module-level OUTPUT_DIR constant; redirect it into
    # tmp_path for test isolation rather than polluting the real output/ dir.
    import ai_video_editor.cli.commands.full as full_module

    monkeypatch.setattr(full_module, "OUTPUT_DIR", tmp_path)

    result = runner.invoke(
        app,
        [
            "full",
            str(smallest_real_clip.parent),
            "--limit", "1",
            "--skip-subtitles",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "scores.json").exists()
    assert (tmp_path / "edit_plan.json").exists()
    assert (tmp_path / "timeline.otio").exists()
    assert (tmp_path / "project.kdenlive").exists()
    assert "Pipeline Complete" in result.output


def test_full_missing_input_dir_exits_nonzero(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["full", str(tmp_path / "does_not_exist")])
    assert result.exit_code != 0
