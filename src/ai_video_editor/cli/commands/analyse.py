"""`video-editor analyse` -- run vision + audio (+ optional GPS) analysis over a clip directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.cli.pipeline import analyse_clip_audio, discover_clips, load_gps
from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.highlight_detection import detect_highlights

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"


@app.command()
def analyse(
    input_dir: Annotated[
        Path, typer.Argument(help="Directory of raw video clips to analyse.")
    ] = PROJECT_ROOT / "assets" / "raw",
    gps: Annotated[
        Path | None, typer.Option(help="Optional GPX/FIT file for ride telemetry.")
    ] = None,
    cache_dir: Annotated[
        Path, typer.Option(help="Where to write per-clip analysis JSON.")
    ] = DEFAULT_CACHE_DIR,
    limit: Annotated[
        int | None,
        typer.Option(help="Only analyse the N smallest clips (useful for smoke-testing)."),
    ] = None,
    max_clip_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-clip-seconds",
            help="Only analyse the first N seconds of each clip. Bounds runtime on long real "
            "footage for fast iteration, independent of --limit's clip count.",
        ),
    ] = None,
) -> None:
    """Run vision (shots + highlights) and audio (silence/speech) analysis over every clip.

    Results are cached per-clip as JSON under `cache_dir` so `score`/`full`
    can reuse them instead of re-decoding video.
    """
    config = load_config()
    clips = discover_clips(input_dir, limit=limit)
    if not clips:
        console.print(f"[yellow]No video clips found under {input_dir}[/yellow]")
        raise typer.Exit(code=1)

    gps_metrics, gps_events = load_gps(gps)
    if gps is not None:
        console.print(f"[green]GPS loaded:[/green] {len(gps_metrics.samples) if gps_metrics else 0} samples")
    else:
        console.print("[dim]No GPS file given -- GPS signals will be skipped.[/dim]")

    cache_dir.mkdir(parents=True, exist_ok=True)

    for clip in track(clips, description="Analysing clips..."):
        highlights = detect_highlights(
            clip,
            scene_threshold=config.scene_detection.threshold,
            min_shot_duration=config.scene_detection.min_shot_duration,
            max_seconds=max_clip_seconds,
        )
        silence, speech = analyse_clip_audio(clip)

        result = {
            "clip": str(clip),
            "shots": [
                {
                    "shot_index": h.shot.index,
                    "start_seconds": h.shot.start_seconds,
                    "end_seconds": h.shot.end_seconds,
                    "highlight_score": h.highlight_score,
                    "scene_category": h.scene_category.value,
                }
                for h in highlights
            ],
            "silence_intervals": [{"start": s.start, "end": s.end} for s in silence],
            "speech_segments": [{"start": s.start, "end": s.end} for s in speech],
        }
        out_path = cache_dir / f"{clip.stem}.analysis.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"  [cyan]{clip.name}[/cyan]: {len(highlights)} shots -> {out_path.name}")

    console.print(f"[bold green]Analysis complete[/bold green] for {len(clips)} clip(s). Cached in {cache_dir}")
