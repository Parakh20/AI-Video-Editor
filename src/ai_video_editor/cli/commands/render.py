"""`video-editor render` -- render a final video from an edit plan."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.gps.metrics import compute_metrics
from ai_video_editor.gps.overlays import (
    build_elevation_marker_positions,
    build_route_map_marker_positions,
    build_timesynced_overlay,
    render_elevation_profile,
    render_marker_dot,
    render_route_map,
)
from ai_video_editor.gps.parsers import load_track
from ai_video_editor.render.presets import RenderPreset, get_preset
from ai_video_editor.render.renderer import GraphicOverlaySpec, MarkerOverlaySpec, RenderResult, render_edit_plan
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EDIT_PLAN = PROJECT_ROOT / "output" / "edit_plan.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "final_render.mp4"


def load_edit_plan_json(path: Path) -> list[EditAction]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EditAction.model_validate(item) for item in raw]


def _rendered_output_duration(kept: list[EditAction], crossfade_s: float) -> float:
    """Approximate the rendered output duration from the kept plan.

    Mirrors the renderer's compositing: kept durations summed, minus one
    crossfade overlap per crossfade transition.
    """
    if not kept:
        return 0.0
    duration = sum(a.duration_seconds for a in kept)
    for action in kept[1:]:
        if action.transition_in == "crossfade":
            duration -= crossfade_s
    return max(duration, 0.0)


def _gps_overlay_fragment(
    gps_path: Path | None, kept: list[EditAction], crossfade_s: float
) -> str | None:
    """Build a time-synced telemetry overlay spanning the rendered output.

    Live speed/distance/gradient are burned in and updated continuously,
    with the full ride mapped linearly across the (shorter) highlight reel --
    see :func:`ai_video_editor.gps.overlays.build_timesynced_overlay`.
    """
    if gps_path is None:
        return None
    track = load_track(gps_path)
    metrics = compute_metrics(track)
    if not metrics.samples:
        return None
    output_duration = _rendered_output_duration(kept, crossfade_s)
    return build_timesynced_overlay(metrics, output_duration)


def _gps_graphic_overlays(
    gps_path: Path | None,
    kept: list[EditAction],
    crossfade_s: float,
    preset: RenderPreset,
    work_dir: Path,
) -> list[GraphicOverlaySpec]:
    """Build the elevation-profile and route-map graphic overlays.

    Renders the elevation-profile strip, route-map plot, and marker dot
    into ``work_dir`` and returns the composited overlay specs -- empty
    when GPS data is unavailable, mirroring :func:`_gps_overlay_fragment`.
    """
    if gps_path is None:
        return []
    track = load_track(gps_path)
    metrics = compute_metrics(track)
    if not metrics.samples:
        return []

    output_duration = _rendered_output_duration(kept, crossfade_s)
    margin = 20
    specs: list[GraphicOverlaySpec] = []

    dot = render_marker_dot(work_dir / "marker_dot.png", radius_px=8)
    dot_radius = dot.width / 2

    elevation_image = render_elevation_profile(metrics, work_dir / "elevation_profile.png")
    if elevation_image is not None and elevation_image.geometry is not None:
        elev_x = (preset.width - elevation_image.width) // 2
        elev_y = preset.height - elevation_image.height - margin
        windows = build_elevation_marker_positions(metrics, elevation_image.geometry, output_duration)
        centered = [
            dataclasses.replace(w, x_px=w.x_px - dot_radius, y_px=w.y_px - dot_radius) for w in windows
        ]
        marker = MarkerOverlaySpec(image_path=dot.path, windows=centered) if centered else None
        specs.append(GraphicOverlaySpec(image_path=elevation_image.path, x=elev_x, y=elev_y, marker=marker))

    route_image = render_route_map(metrics, work_dir / "route_map.png")
    if route_image is not None and route_image.geometry is not None:
        route_x = preset.width - route_image.width - margin
        route_y = preset.height - route_image.height - margin
        windows = build_route_map_marker_positions(metrics, route_image.geometry, output_duration)
        centered = [
            dataclasses.replace(w, x_px=w.x_px - dot_radius, y_px=w.y_px - dot_radius) for w in windows
        ]
        marker = MarkerOverlaySpec(image_path=dot.path, windows=centered) if centered else None
        specs.append(GraphicOverlaySpec(image_path=route_image.path, x=route_x, y=route_y, marker=marker))

    return specs


def run_render(
    edit_plan_path: Path,
    *,
    preset_name: str,
    output_path: Path,
    gps: Path | None,
    crossfade_s: float,
    dry_run: bool,
) -> RenderResult | None:
    """Shared by the `render` and `full` commands."""
    actions = load_edit_plan_json(edit_plan_path)
    preset = get_preset(preset_name)

    kept_actions = [a for a in actions if a.kind == "keep"]
    kept = len(kept_actions)
    if kept == 0:
        raise ValueError(f"Edit plan at {edit_plan_path} has no kept segments to render")

    overlay = _gps_overlay_fragment(gps, kept_actions, crossfade_s)

    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] would render {kept} segment(s) at preset "
            f"'{preset.name}' ({preset.width}x{preset.height}@{preset.fps}) -> {output_path}"
            + (" with GPS overlay" if overlay else "")
        )
        return None

    graphic_overlays = _gps_graphic_overlays(gps, kept_actions, crossfade_s, preset, output_path.parent)

    result = render_edit_plan(
        actions,
        output_path,
        preset,
        crossfade_s=crossfade_s,
        overlay_drawtext=overlay,
        graphic_overlays=graphic_overlays,
        prefer_gpu=True,
    )
    return result


@app.command()
def render(
    edit_plan: Annotated[Path, typer.Option(help="edit_plan.json to render.")] = DEFAULT_EDIT_PLAN,
    preset: Annotated[
        str | None, typer.Option(help="Render preset: 1080p, 1440p, 4k, youtube, instagram_reel, youtube_shorts.")
    ] = None,
    output: Annotated[Path, typer.Option(help="Output video path.")] = DEFAULT_OUTPUT,
    gps: Annotated[Path | None, typer.Option(help="Optional GPX/FIT file for a burned-in stat overlay.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Plan the render without actually encoding.")] = False,
) -> None:
    """Render the kept segments of an edit plan to a final video file."""
    if not edit_plan.exists():
        console.print(f"[red]Edit plan not found at {edit_plan}[/red] -- run `video-editor timeline` first.")
        raise typer.Exit(code=1)

    config = load_config()
    resolved_preset = preset or config.render.default_preset

    result = run_render(
        edit_plan,
        preset_name=resolved_preset,
        output_path=output,
        gps=gps,
        crossfade_s=config.render.crossfade_seconds,
        dry_run=dry_run,
    )
    if result is not None:
        console.print(
            f"[bold green]Rendered[/bold green] {result.clip_count} clip(s) via "
            f"{result.used_encoder} -> {result.output_path}"
        )
