"""Cycling telemetry overlays: drawtext filters + rendered PNG graphics (Phase 11).

Every public function here takes ``metrics: RideMetrics | None`` and returns
``None`` (or an empty/no-op result) when GPS data is unavailable — the real
footage in this project has no GPS track, so this is the common case, not
an edge case. Text overlays are built as ffmpeg ``drawtext`` fragments
(time-varying via ``expansion``); graphical overlays (elevation profile,
route map, stats card) are rendered once to a transparent PNG with
matplotlib/PIL, since neither needs ImageMagick.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from ai_video_editor.gps.models import RideMetrics
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_MPS_TO_KMH = 3.6


def _interpolate(metrics: RideMetrics, elapsed_s: float, attr: str) -> float | None:
    """Linearly interpolate a numeric metric sample field at ``elapsed_s``."""
    samples = metrics.samples
    if not samples:
        return None
    if elapsed_s <= samples[0].elapsed_s:
        return getattr(samples[0], attr)
    if elapsed_s >= samples[-1].elapsed_s:
        return getattr(samples[-1], attr)

    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid].elapsed_s <= elapsed_s:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    va, vb = getattr(a, attr), getattr(b, attr)
    if va is None or vb is None:
        return va if va is not None else vb
    span = b.elapsed_s - a.elapsed_s
    if span <= 0:
        return va
    t = (elapsed_s - a.elapsed_s) / span
    return va + (vb - va) * t


def speed_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a ``"NN km/h"`` label for the sample nearest ``elapsed_s``, or None."""
    if metrics is None or not metrics.samples:
        return None
    speed = _interpolate(metrics, elapsed_s, "speed_mps")
    if speed is None:
        return None
    return f"{speed * _MPS_TO_KMH:.0f} km/h"


def distance_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a ``"NN.N km"`` cumulative-distance label, or None."""
    if metrics is None or not metrics.samples:
        return None
    dist = _interpolate(metrics, elapsed_s, "distance_m")
    if dist is None:
        return None
    return f"{dist / 1000:.1f} km"


def gradient_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a signed gradient percentage label, or None."""
    if metrics is None or not metrics.samples:
        return None
    grad = _interpolate(metrics, elapsed_s, "gradient_pct")
    if grad is None:
        return None
    return f"{grad:+.1f}%"


def power_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a ``"NNN W"`` power label, or None if unavailable at this instant."""
    if metrics is None or not metrics.samples:
        return None
    power = _interpolate(metrics, elapsed_s, "power_w")
    if power is None:
        return None
    return f"{power:.0f} W"


def heart_rate_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a ``"NNN bpm"`` heart-rate label, or None if unavailable."""
    if metrics is None or not metrics.samples:
        return None
    hr = _interpolate(metrics, elapsed_s, "heart_rate_bpm")
    if hr is None:
        return None
    return f"{hr:.0f} bpm"


def cadence_overlay_text(metrics: RideMetrics | None, elapsed_s: float) -> str | None:
    """Return a ``"NN rpm"`` cadence label, or None if unavailable."""
    if metrics is None or not metrics.samples:
        return None
    cad = _interpolate(metrics, elapsed_s, "cadence_rpm")
    if cad is None:
        return None
    return f"{cad:.0f} rpm"


def drawtext_filter(
    text: str,
    *,
    x: str = "40",
    y: str = "40",
    font_size: int = 36,
    font_color: str = "white",
    box: bool = True,
) -> str:
    """Return a ``drawtext`` fragment burning a static label onto video.

    For time-varying overlays composed frame-by-frame in a full render,
    prefer the PNG-sequence overlay helpers below; this is for a single
    static label (e.g. a title card or a fixed stat in a smoke test).
    """
    escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    box_part = ":box=1:boxcolor=black@0.5:boxborderw=10" if box else ""
    # expansion=none: our labels are literal (e.g. "+6.0%", which otherwise
    # trips ffmpeg's `%{...}` text-expansion parser with a "Stray %" warning).
    return (
        f"drawtext=text='{escaped}':expansion=none:x={x}:y={y}:"
        f"fontsize={font_size}:fontcolor={font_color}{box_part}"
    )


#: Ordered field name -> text-builder mapping for the time-synced overlay.
#: Each builder returns a formatted label (or None when unavailable at that
#: instant); the composed overlay line joins whichever fields are present.
_OVERLAY_FIELD_BUILDERS: dict[str, Callable[[RideMetrics, float], str | None]] = {
    "speed": speed_overlay_text,
    "distance": distance_overlay_text,
    "gradient": gradient_overlay_text,
    "power": power_overlay_text,
    "heart_rate": heart_rate_overlay_text,
    "cadence": cadence_overlay_text,
}

_DEFAULT_OVERLAY_FIELDS: tuple[str, ...] = ("speed", "distance", "gradient")
_DEFAULT_MAX_WINDOWS = 600


@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float
    ride_elapsed_s: float
    """The point on the ride's own timeline that this output window maps to."""


def _time_windows(
    output_duration_s: float,
    ride_duration_s: float,
    update_interval_s: float,
    max_windows: int,
) -> list[TimeWindow]:
    """Divide ``output_duration_s`` into windows, each mapped onto ride time.

    Shared by the text overlay and the graphical markers so window
    boundaries stay identical across both. The full ride is mapped
    linearly onto the output timeline: window ``k`` starting at output time
    ``start`` maps to ride-elapsed time ``(start / output_duration_s) *
    ride_duration_s``.
    """
    window_count = math.ceil(output_duration_s / update_interval_s)
    if window_count > max_windows:
        window_count = max_windows
    dt = output_duration_s / window_count

    windows: list[TimeWindow] = []
    for k in range(window_count):
        start = k * dt
        end = output_duration_s if k == window_count - 1 else (k + 1) * dt
        ride_elapsed = (start / output_duration_s) * ride_duration_s
        windows.append(TimeWindow(start_s=start, end_s=end, ride_elapsed_s=ride_elapsed))
    return windows


def _overlay_line(metrics: RideMetrics, ride_elapsed_s: float, fields: Sequence[str]) -> str:
    """Compose a single overlay line from the available telemetry fields."""
    parts: list[str] = []
    for field in fields:
        builder = _OVERLAY_FIELD_BUILDERS.get(field)
        if builder is None:
            continue
        label = builder(metrics, ride_elapsed_s)
        if label is not None:
            parts.append(label)
    return "   ".join(parts)


def build_timesynced_overlay(
    metrics: RideMetrics | None,
    output_duration_s: float,
    *,
    fields: Sequence[str] = _DEFAULT_OVERLAY_FIELDS,
    update_interval_s: float = 1.0,
    max_windows: int = _DEFAULT_MAX_WINDOWS,
    x: str = "40",
    y: str = "40",
    font_size: int = 36,
    font_color: str = "white",
) -> str | None:
    """Build a continuously-updating telemetry overlay as an ffmpeg filterchain.

    Unlike :func:`drawtext_filter` (a single static label), this burns in
    live-updating ride telemetry. The full ride is mapped linearly onto the
    rendered output timeline -- a highlight reel is shorter than the ride and
    has no true frame-accurate GPS sync, so the whole ride is *stretched*
    across ``output_duration_s`` so the telemetry sweeps the entire ride as
    the reel plays.

    The result is a comma-joined chain of ``drawtext`` filters, one per
    update window, each gated with ``enable='between(t,A,B)'`` so only the
    window covering the current output timestamp is drawn. Returns ``None``
    when GPS data is unavailable, the output has no duration, or no field
    yields any text.

    ``update_interval_s`` controls how often the label changes; the number of
    windows is capped at ``max_windows`` (the interval is widened as needed)
    to keep the filter graph bounded on long outputs.
    """
    if metrics is None or not metrics.samples:
        return None
    if output_duration_s <= 0:
        return None
    if update_interval_s <= 0:
        raise ValueError("update_interval_s must be positive")

    ride_duration_s = metrics.duration_s
    windows = _time_windows(output_duration_s, ride_duration_s, update_interval_s, max_windows)

    fragments: list[str] = []
    for w in windows:
        line = _overlay_line(metrics, w.ride_elapsed_s, fields)
        if not line:
            continue
        fragment = drawtext_filter(
            line, x=x, y=y, font_size=font_size, font_color=font_color
        )
        fragments.append(f"{fragment}:enable='between(t,{w.start_s:.3f},{w.end_s:.3f})'")

    if not fragments:
        return None
    return ",".join(fragments)


@dataclass(frozen=True)
class OverlayGeometry:
    """Maps a matplotlib axes' data space onto its rendered image's pixels."""

    x_data_range: tuple[float, float]
    y_data_range: tuple[float, float]
    bbox_px: tuple[float, float, float, float]
    """(left, top, right, bottom) in top-left-origin image-pixel space."""


def _data_to_pixel(x: float, y: float, geometry: OverlayGeometry) -> tuple[float, float]:
    """Map a data-space point to image-pixel coordinates via ``geometry``.

    Falls back to the bbox center on a degenerate (zero-width or
    zero-height) data range, to avoid a divide-by-zero.
    """
    x_lo, x_hi = geometry.x_data_range
    y_lo, y_hi = geometry.y_data_range
    left, top, right, bottom = geometry.bbox_px

    x_span = x_hi - x_lo
    if x_span == 0:
        px = (left + right) / 2
    else:
        px = left + ((x - x_lo) / x_span) * (right - left)

    y_span = y_hi - y_lo
    if y_span == 0:
        py = (top + bottom) / 2
    else:
        # Data y increases upward; image y increases downward.
        py = bottom - ((y - y_lo) / y_span) * (bottom - top)

    return px, py


@dataclass(frozen=True)
class OverlayImage:
    path: Path
    width: int
    height: int
    geometry: OverlayGeometry | None = None


def render_elevation_profile(
    metrics: RideMetrics | None,
    output_path: Path,
    *,
    width_px: int = 800,
    height_px: int = 200,
) -> OverlayImage | None:
    """Render a transparent-background elevation-vs-distance strip to PNG.

    Returns None (writes nothing) when ``metrics`` is None/empty or has no
    elevation data at all.
    """
    if metrics is None or not metrics.samples:
        logger.debug("render_elevation_profile: no GPS metrics available, skipping")
        return None
    elevations = [s.elevation_m for s in metrics.samples]
    if all(e is None for e in elevations):
        logger.debug("render_elevation_profile: no elevation data in track, skipping")
        return None

    distances_km = [s.distance_m / 1000 for s in metrics.samples]
    filled = [e if e is not None else 0.0 for e in elevations]

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.fill_between(distances_km, filled, color="white", alpha=0.35)
    ax.plot(distances_km, filled, color="white", linewidth=2)
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    geometry = OverlayGeometry(
        x_data_range=ax.get_xlim(),
        y_data_range=ax.get_ylim(),
        bbox_px=(bbox.x0, height_px - bbox.y1, bbox.x1, height_px - bbox.y0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, dpi=dpi)
    plt.close(fig)

    logger.info("render_elevation_profile: wrote %s", output_path)
    return OverlayImage(path=output_path, width=width_px, height=height_px, geometry=geometry)


def render_route_map(
    metrics: RideMetrics | None,
    output_path: Path,
    *,
    size_px: int = 300,
) -> OverlayImage | None:
    """Render a simple lat/lon route-shape overlay (no map tiles, no network).

    Returns None when GPS data is unavailable.
    """
    if metrics is None or not metrics.samples:
        logger.debug("render_route_map: no GPS metrics available, skipping")
        return None

    lats = [s.lat for s in metrics.samples]
    lons = [s.lon for s in metrics.samples]

    dpi = 100
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.plot(lons, lats, color="white", linewidth=2.5)
    ax.plot(lons[0], lats[0], "o", color="lime", markersize=6)
    ax.plot(lons[-1], lats[-1], "o", color="red", markersize=6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    geometry = OverlayGeometry(
        x_data_range=ax.get_xlim(),
        y_data_range=ax.get_ylim(),
        bbox_px=(bbox.x0, size_px - bbox.y1, bbox.x1, size_px - bbox.y0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, dpi=dpi)
    plt.close(fig)

    logger.info("render_route_map: wrote %s", output_path)
    return OverlayImage(path=output_path, width=size_px, height=size_px, geometry=geometry)


def render_ride_stats_card(
    metrics: RideMetrics | None,
    output_path: Path,
    *,
    width_px: int = 500,
    height_px: int = 260,
) -> OverlayImage | None:
    """Render a PIL-drawn summary card (distance/time/avg speed/elevation gain).

    Returns None when GPS data is unavailable.
    """
    if metrics is None or not metrics.samples:
        logger.debug("render_ride_stats_card: no GPS metrics available, skipping")
        return None

    speeds = [s.speed_mps for s in metrics.samples]
    avg_speed_kmh = (sum(speeds) / len(speeds)) * _MPS_TO_KMH if speeds else 0.0
    elevations = [s.elevation_m for s in metrics.samples if s.elevation_m is not None]
    elevation_gain_m = 0.0
    for prev, cur in zip(elevations, elevations[1:], strict=False):
        if cur > prev:
            elevation_gain_m += cur - prev

    lines = [
        "Ride Summary",
        f"Distance: {metrics.total_distance_m / 1000:.1f} km",
        f"Duration: {metrics.duration_s / 60:.0f} min",
        f"Avg speed: {avg_speed_kmh:.1f} km/h",
        f"Elevation gain: {elevation_gain_m:.0f} m",
    ]

    img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 140))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 20
    for i, line in enumerate(lines):
        draw.text((20, y), line, fill=(255, 255, 255, 255), font=font)
        y += 24 if i > 0 else 32

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    logger.info("render_ride_stats_card: wrote %s", output_path)
    return OverlayImage(path=output_path, width=width_px, height=height_px)
