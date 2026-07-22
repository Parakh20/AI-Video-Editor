"""Every command must expose working --help text."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ai_video_editor.cli.main import app

COMMANDS = ["analyse", "score", "subtitles", "timeline", "render", "full"]


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_exits_cleanly(runner: CliRunner, command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert command in result.output.lower() or "usage" in result.output.lower()


def test_app_help_lists_all_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
