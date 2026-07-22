"""`video-editor score` -- score every shot in a clip directory into scores.json."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from ai_video_editor.analysis.scoring import SegmentScore, write_scores_json
from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.cli.pipeline import discover_clips, load_gps, load_music, score_clip
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "scores.json"


def run_scoring(
    input_dir: Path,
    *,
    gps: Path | None,
    music: Path | None,
    limit: int | None,
    max_seconds_per_clip: float | None = None,
) -> list[SegmentScore]:
    """Score every clip under `input_dir`. Shared by the `score` and `full` commands."""
    config = load_config()
    clips = discover_clips(input_dir, limit=limit)
    if not clips:
        raise FileNotFoundError(f"No video clips found under {input_dir}")

    gps_metrics, gps_events = load_gps(gps)
    music_analysis = load_music(music)

    all_scores: list[SegmentScore] = []
    for clip in track(clips, description="Scoring clips..."):
        clip_scores = score_clip(
            clip,
            weights=config.scoring,
            scene_threshold=config.scene_detection.threshold,
            min_shot_duration=config.scene_detection.min_shot_duration,
            max_seconds=max_seconds_per_clip,
            gps_metrics=gps_metrics,
            gps_events=gps_events,
            music=music_analysis,
        )
        all_scores.extend(clip_scores)
        console.print(f"  [cyan]{clip.name}[/cyan]: {len(clip_scores)} shot(s) scored")

    return all_scores


@app.command()
def score(
    input_dir: Annotated[
        Path, typer.Argument(help="Directory of raw video clips to score.")
    ] = PROJECT_ROOT / "assets" / "raw",
    gps: Annotated[Path | None, typer.Option(help="Optional GPX/FIT ride telemetry.")] = None,
    music: Annotated[Path | None, typer.Option(help="Optional soundtrack for music-sync scoring.")] = None,
    output: Annotated[Path, typer.Option(help="Where to write scores.json.")] = DEFAULT_OUTPUT,
    limit: Annotated[int | None, typer.Option(help="Only score the N smallest clips.")] = None,
    max_clip_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-clip-seconds",
            help="Only analyse the first N seconds of each clip. Full per-shot vision+audio "
            "analysis on long real footage can take minutes per clip; this bounds that for "
            "fast iteration (e.g. --max-clip-seconds 20) without touching --limit's clip count.",
        ),
    ] = None,
) -> None:
    """Score every shot across the input clips (visual quality, motion, GPS/audio/music signals)."""
    scores = run_scoring(
        input_dir, gps=gps, music=music, limit=limit, max_seconds_per_clip=max_clip_seconds
    )
    if not scores:
        console.print("[yellow]No shots were scored (no usable clips/shots found).[/yellow]")
        raise typer.Exit(code=1)

    write_scores_json(scores, output)
    mean_score = sum(s.overall_score for s in scores) / len(scores)
    console.print(
        f"[bold green]Wrote {len(scores)} scored segment(s)[/bold green] to {output} "
        f"(mean score={mean_score:.1f})"
    )
