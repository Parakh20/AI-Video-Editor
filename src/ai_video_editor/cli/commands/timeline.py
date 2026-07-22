"""`video-editor timeline` -- build the edit plan, timeline.otio, and project.kdenlive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ai_video_editor.analysis.scoring import SegmentScore
from ai_video_editor.audio.beat_detection import BeatGrid, detect_beats
from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.editing.decision_engine import EditAction, build_edit_plan, write_edit_plan_json
from ai_video_editor.timeline.kdenlive_export import write_kdenlive_project
from ai_video_editor.timeline.otio_builder import build_otio_timeline, write_otio
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCORES = PROJECT_ROOT / "output" / "scores.json"
DEFAULT_EDIT_PLAN = PROJECT_ROOT / "output" / "edit_plan.json"
DEFAULT_OTIO = PROJECT_ROOT / "output" / "timeline.otio"
DEFAULT_KDENLIVE = PROJECT_ROOT / "output" / "project.kdenlive"


def load_scores_json(path: Path) -> list[SegmentScore]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SegmentScore.model_validate(item) for item in raw]


def build_timeline_outputs(
    scores: list[SegmentScore],
    *,
    beat_grid: BeatGrid | None,
    threshold: float,
    merge_gap_tolerance: float,
    beat_tolerance: float,
    trim_seconds: float,
    edit_plan_path: Path,
    otio_path: Path,
    kdenlive_path: Path,
) -> list[EditAction]:
    """Build the edit plan and both timeline representations. Shared by `timeline` and `full`."""
    actions = build_edit_plan(
        scores,
        threshold=threshold,
        beat_grid=beat_grid,
        merge_gap_tolerance=merge_gap_tolerance,
        beat_tolerance=beat_tolerance,
        trim_seconds=trim_seconds,
    )
    write_edit_plan_json(actions, edit_plan_path)

    otio_timeline = build_otio_timeline(actions, song_path=None)
    write_otio(otio_timeline, otio_path)
    write_kdenlive_project(actions, kdenlive_path)
    return actions


@app.command()
def timeline(
    scores: Annotated[Path, typer.Option(help="scores.json produced by `score`.")] = DEFAULT_SCORES,
    music: Annotated[
        Path | None, typer.Option(help="Soundtrack to derive a beat grid from, for beat-aligned cuts.")
    ] = None,
    edit_plan_out: Annotated[Path, typer.Option(help="Where to write edit_plan.json.")] = DEFAULT_EDIT_PLAN,
    otio_out: Annotated[Path, typer.Option(help="Where to write timeline.otio.")] = DEFAULT_OTIO,
    kdenlive_out: Annotated[Path, typer.Option(help="Where to write project.kdenlive.")] = DEFAULT_KDENLIVE,
) -> None:
    """Build an explained edit plan from scores.json, then export timeline.otio + project.kdenlive."""
    if not scores.exists():
        console.print(f"[red]scores.json not found at {scores}[/red] -- run `video-editor score` first.")
        raise typer.Exit(code=1)

    config = load_config()
    segment_scores = load_scores_json(scores)

    beat_grid: BeatGrid | None = None
    if music is not None:
        beat_grid = detect_beats(music)
        console.print(f"[green]Beat grid:[/green] {len(beat_grid.beat_times)} beats, {beat_grid.tempo_bpm:.1f} BPM")
    else:
        console.print("[dim]No music given -- cuts will not be beat-aligned.[/dim]")

    actions = build_timeline_outputs(
        segment_scores,
        beat_grid=beat_grid,
        threshold=config.decision.score_threshold,
        merge_gap_tolerance=config.decision.merge_gap_tolerance,
        beat_tolerance=config.decision.beat_tolerance,
        trim_seconds=config.decision.trim_seconds,
        edit_plan_path=edit_plan_out,
        otio_path=otio_out,
        kdenlive_path=kdenlive_out,
    )
    kept = sum(1 for a in actions if a.kind == "keep")
    console.print(
        f"[bold green]Edit plan built:[/bold green] {kept} kept / {len(actions) - kept} dropped "
        f"-> {edit_plan_out}, {otio_out}, {kdenlive_out}"
    )
