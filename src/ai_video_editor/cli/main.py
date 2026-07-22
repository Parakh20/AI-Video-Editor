"""Entry point for the `video-editor` command line interface.

Commands live in `ai_video_editor.cli.commands.*` and register themselves
onto `app` when imported below. `app` is imported by name from
`pyproject.toml`'s `[project.scripts]` entry point.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="video-editor",
    help="AI-powered production pipeline for cinematic cycling POV videos.",
    no_args_is_help=True,
)

# Each import below attaches a command to `app` via `@app.command()`. Import
# order doesn't matter; `app` already exists in this module by this point,
# so the circular `from ai_video_editor.cli.main import app` in each command
# module resolves against the (already-partially-initialized) module object.
from ai_video_editor.cli.commands import analyse  # noqa: E402,F401
from ai_video_editor.cli.commands import full  # noqa: E402,F401
from ai_video_editor.cli.commands import render  # noqa: E402,F401
from ai_video_editor.cli.commands import score  # noqa: E402,F401
from ai_video_editor.cli.commands import subtitles  # noqa: E402,F401
from ai_video_editor.cli.commands import timeline  # noqa: E402,F401

__all__ = ["app"]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
