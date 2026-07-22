"""`video-editor full` -- the end-to-end pipeline: analyse -> score -> decide ->
subtitles -> timeline (OTIO + Kdenlive) -> render.

Each stage writes its output to disk before the next stage runs, so a
failure partway through never loses earlier work -- `scores.json`,
`edit_plan.json`, subtitle files, `timeline.otio`, and `project.kdenlive`
are all real artifacts on disk the moment their stage completes, not
transient in-memory state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ai_video_editor.analysis.scoring import write_scores_json
from ai_video_editor.audio.beat_detection import BeatGrid, detect_beats
from ai_video_editor.cli.commands.render import run_render
from ai_video_editor.cli.commands.subtitles import generate_subtitles_for
from ai_video_editor.cli.commands.timeline import build_timeline_outputs
from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.cli.pipeline import discover_clips
from ai_video_editor.cli.commands.score import run_scoring
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = PROJECT_ROOT / "output"


@app.command()
def full(
    input_dir: Annotated[
        Path, typer.Argument(help="Directory of raw video clips to edit.")
    ] = PROJECT_ROOT / "assets" / "raw",
    music: Annotated[Path | None, typer.Option(help="Soundtrack for beat-synced cuts.")] = None,
    gps: Annotated[Path | None, typer.Option(help="Optional GPX/FIT ride telemetry.")] = None,
    preset: Annotated[str | None, typer.Option(help="Render preset (default from config: 1080p).")] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            help="Only process the N smallest input clips. Strongly recommended for a first "
            "run / smoke test -- full-length real footage can take a long time to fully "
            "analyse and render."
        ),
    ] = None,
    max_clip_seconds: Annotated[
        float | None,
        typer.Option(
            "--max-clip-seconds",
            help="Only analyse the first N seconds of each clip (vision+audio scoring stage). "
            "Bounds runtime on long real footage for fast iteration, independent of --limit's "
            "clip count -- e.g. `--limit 2 --max-clip-seconds 20` scores 2 clips, ~20s each, "
            "instead of their full length.",
        ),
    ] = None,
    skip_subtitles: Annotated[bool, typer.Option(help="Skip Whisper transcription.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Plan every stage but skip the final encode.")] = False,
) -> None:
    """Run the complete pipeline: analyse -> score -> decide -> subtitles -> timeline -> render.

    This is the single command referenced by the project's success
    criterion: `video-editor full ./assets/raw`.
    """
    config = load_config()
    clips = discover_clips(input_dir, limit=limit)
    if not clips:
        console.print(f"[red]No video clips found under {input_dir}[/red]")
        raise typer.Exit(code=1)

    warnings: list[str] = []

    # --- Stage 1+2: analyse & score --------------------------------------
    console.rule("[bold]Stage 1-2: Analyse + Score[/bold]")
    scores = run_scoring(
        input_dir, gps=gps, music=music, limit=limit, max_seconds_per_clip=max_clip_seconds
    )
    if not scores:
        console.print("[red]No shots were scored -- aborting.[/red]")
        raise typer.Exit(code=1)
    scores_path = OUTPUT_DIR / "scores.json"
    write_scores_json(scores, scores_path)
    if gps is None:
        warnings.append("No GPS data provided -- GPS scoring signals and overlays were skipped.")

    # --- Stage 3: beat grid + edit decision -------------------------------
    console.rule("[bold]Stage 3: Edit Decision[/bold]")
    beat_grid: BeatGrid | None = None
    if music is not None:
        beat_grid = detect_beats(music)
        console.print(f"Beat grid: {len(beat_grid.beat_times)} beats, {beat_grid.tempo_bpm:.1f} BPM")
    else:
        warnings.append("No music provided -- cuts were not beat-aligned.")

    edit_plan_path = OUTPUT_DIR / "edit_plan.json"
    otio_path = OUTPUT_DIR / "timeline.otio"
    kdenlive_path = OUTPUT_DIR / "project.kdenlive"
    actions = build_timeline_outputs(
        scores,
        beat_grid=beat_grid,
        threshold=config.decision.score_threshold,
        merge_gap_tolerance=config.decision.merge_gap_tolerance,
        beat_tolerance=config.decision.beat_tolerance,
        trim_seconds=config.decision.trim_seconds,
        edit_plan_path=edit_plan_path,
        otio_path=otio_path,
        kdenlive_path=kdenlive_path,
    )
    kept = [a for a in actions if a.kind == "keep"]
    dropped = len(actions) - len(kept)
    if not kept:
        console.print(
            "[red]Every segment was dropped by the edit decision engine -- "
            "nothing to render. Try lowering decision.score_threshold in configs/default.yaml.[/red]"
        )
        raise typer.Exit(code=1)

    # --- Stage 4: subtitles ------------------------------------------------
    subtitle_count = 0
    if not skip_subtitles:
        console.rule("[bold]Stage 4: Subtitles[/bold]")
        subs_dir = OUTPUT_DIR / "subtitles"
        results = generate_subtitles_for(
            input_dir, output_dir=subs_dir, model_size=config.subtitles.whisper_model_size, limit=limit
        )
        subtitle_count = sum(len(r.segments) for r in results.values())
    else:
        warnings.append("Subtitles skipped (--skip-subtitles).")

    # --- Stage 5: render -----------------------------------------------
    console.rule("[bold]Stage 5: Render[/bold]")
    resolved_preset = preset or config.render.default_preset
    output_path = OUTPUT_DIR / f"final_render_{resolved_preset}.mp4"
    render_result = run_render(
        edit_plan_path,
        preset_name=resolved_preset,
        output_path=output_path,
        gps=gps,
        crossfade_s=config.render.crossfade_seconds,
        dry_run=dry_run,
    )

    # --- Summary -------------------------------------------------------
    console.rule("[bold green]Pipeline Complete[/bold green]")
    table = Table(show_header=False)
    table.add_row("Clips processed", str(len(clips)))
    table.add_row("Shots scored", str(len(scores)))
    table.add_row("Segments kept / dropped", f"{len(kept)} / {dropped}")
    table.add_row("Subtitle segments", str(subtitle_count) if not skip_subtitles else "skipped")
    table.add_row("scores.json", str(scores_path))
    table.add_row("edit_plan.json", str(edit_plan_path))
    table.add_row("timeline.otio", str(otio_path))
    table.add_row("project.kdenlive", str(kdenlive_path))
    if render_result is not None:
        table.add_row("Final render", f"{render_result.output_path} (encoder={render_result.used_encoder})")
    elif dry_run:
        table.add_row("Final render", "skipped (--dry-run)")
    console.print(table)

    if warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in warnings:
            console.print(f"  - {w}")
