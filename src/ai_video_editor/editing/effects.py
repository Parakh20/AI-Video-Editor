"""FFmpeg-filter-graph effects: speed ramps, transitions, stabilization, grading (Phase 10).

Every function here returns either a plain ffmpeg filter-graph *fragment*
(a string usable inside ``-filter_complex``/``-vf``) or, for two-pass
operations like stabilization, applies the filter via a subprocess call and
returns the output path. Nothing here requires ImageMagick — this machine
does not have it installed, and MoviePy's ``TextClip`` (which needs it) is
avoided project-wide in favor of ffmpeg's native ``drawtext``/``overlay``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ai_video_editor.utils.logging import get_logger
from ai_video_editor.vision.day_night_classifier import LightingCondition

logger = get_logger(__name__)


def speed_ramp_filter(factor: float) -> str:
    """Return a ``setpts`` fragment for a constant speed change.

    ``factor`` > 1.0 speeds the clip up (e.g. 2.0 = double speed);
    < 1.0 slows it down. Audio is not touched here — pair with an
    ``atempo`` chain at the call site if audio speed must match.
    """
    if factor <= 0:
        raise ValueError(f"speed_ramp_filter: factor must be > 0, got {factor}")
    return f"setpts={1.0 / factor:.6f}*PTS"


def crossfade_filter(
    duration_s: float,
    offset_s: float,
    *,
    transition: str = "fade",
) -> str:
    """Return an ``xfade`` fragment for a two-input crossfade.

    ``offset_s`` is the timestamp (in the first input's timeline) at which
    the transition begins; ``duration_s`` is how long it lasts. Caller is
    responsible for labelling the two video inputs feeding this filter.
    """
    return f"xfade=transition={transition}:duration={duration_s:.3f}:offset={offset_s:.3f}"


def fade_filter(
    *,
    fade_type: str,
    start_s: float,
    duration_s: float,
) -> str:
    """Return a ``fade`` fragment (video fade in/out to/from black)."""
    if fade_type not in ("in", "out"):
        raise ValueError(f"fade_filter: fade_type must be 'in' or 'out', got {fade_type!r}")
    return f"fade=t={fade_type}:st={start_s:.3f}:d={duration_s:.3f}"


def zoom_filter(
    *,
    zoom_factor: float,
    duration_frames: int,
    frame_rate: float,
) -> str:
    """Return a ``zoompan`` fragment for a smooth zoom-in over the clip.

    ``zoom_factor`` is the final zoom level reached at the last frame
    (1.0 = no zoom). Pans toward the frame center.
    """
    if zoom_factor < 1.0:
        raise ValueError(f"zoom_filter: zoom_factor must be >= 1.0, got {zoom_factor}")
    zoom_step = (zoom_factor - 1.0) / max(duration_frames, 1)
    return (
        f"zoompan=z='min(zoom+{zoom_step:.8f},{zoom_factor})':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={duration_frames}:fps={frame_rate:.3f}"
    )


def crop_filter(*, width: int, height: int, x: int = -1, y: int = -1) -> str:
    """Return a ``crop`` fragment. ``x``/``y`` of -1 centers the crop."""
    x_expr = "(in_w-out_w)/2" if x < 0 else str(x)
    y_expr = "(in_h-out_h)/2" if y < 0 else str(y)
    return f"crop={width}:{height}:{x_expr}:{y_expr}"


def letterbox_filter(*, target_width: int, target_height: int) -> str:
    """Return a ``pad`` fragment that letterboxes the input to fit a target box.

    Assumes the input has already been scaled to fit within
    ``target_width``x``target_height`` on at least one axis (see
    :func:`ai_video_editor.render.renderer` for how this is composed with
    a preceding ``scale``).
    """
    return (
        f"pad={target_width}:{target_height}:"
        f"(ow-iw)/2:(oh-ih)/2:color=black"
    )


def lut3d_filter(lut_path: str | Path) -> str:
    """Return a ``lut3d`` fragment applying a ``.cube`` LUT file."""
    return f"lut3d=file='{Path(lut_path).resolve()}'"


@dataclass(frozen=True)
class StabilizationResult:
    output_path: Path
    transform_file: Path


def stabilize_clip(
    input_path: Path,
    output_path: Path,
    *,
    shakiness: int = 5,
    smoothing: int = 10,
    ffmpeg_bin: str = "ffmpeg",
) -> StabilizationResult:
    """Run a two-pass vidstab stabilization pass and write the result.

    Pass 1 (``vidstabdetect``) analyzes motion into a transform file; pass 2
    (``vidstabtransform``) applies the correction. Requires ffmpeg built
    with ``--enable-libvidstab`` (confirmed present on this machine).
    """
    transform_file = output_path.with_suffix(".trf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detect_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"vidstabdetect=shakiness={shakiness}:result={transform_file}",
        "-f",
        "null",
        "-",
    ]
    logger.info("stabilize_clip: pass 1 (detect) on %s", input_path)
    subprocess.run(detect_cmd, check=True, capture_output=True, text=True)

    transform_cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        f"vidstabtransform=input={transform_file}:smoothing={smoothing}",
        "-c:a",
        "copy",
        str(output_path),
    ]
    logger.info("stabilize_clip: pass 2 (transform) -> %s", output_path)
    subprocess.run(transform_cmd, check=True, capture_output=True, text=True)

    return StabilizationResult(output_path=output_path, transform_file=transform_file)


class GradeStyle(str, Enum):
    DAY_NATURAL = "day_natural"
    NIGHT_LIFTED = "night_lifted"
    GOLDEN_HOUR = "golden_hour"


@dataclass(frozen=True)
class GradeSuggestion:
    style: GradeStyle
    reason: str
    filter_chain: str


def suggest_color_grade(
    lighting: LightingCondition,
    *,
    warm_ratio: float | None = None,
) -> GradeSuggestion:
    """Heuristically suggest a grading approach + ffmpeg filter chain.

    This is a rule-based recommendation, not a learned model: day footage
    gets natural/vibrant curves, footage with a strong warm channel bias
    near dusk gets a golden-hour push, and night footage gets denoising
    with lifted shadows while protecting highlights (headlights/streetlights
    clip easily if shadows are lifted too aggressively).
    """
    if lighting is LightingCondition.NIGHT:
        return GradeSuggestion(
            style=GradeStyle.NIGHT_LIFTED,
            reason=(
                "night footage: denoise first (sensor noise dominates in low light), "
                "then lift shadows moderately while curving down the highlight roll-off "
                "so streetlights/headlights don't clip to flat white"
            ),
            filter_chain=(
                "hqdn3d=4:3:6:4,"
                "curves=master='0/0 0.2/0.32 0.6/0.68 1/0.95'"
            ),
        )
    if warm_ratio is not None and warm_ratio > 1.15:
        return GradeSuggestion(
            style=GradeStyle.GOLDEN_HOUR,
            reason=(
                f"day footage with strong warm-channel bias (ratio={warm_ratio:.2f}) "
                "suggests golden-hour light; push warmth and contrast rather than "
                "correcting it toward neutral"
            ),
            filter_chain="eq=contrast=1.08:saturation=1.15,colorbalance=rm=0.08:gm=0.02:bm=-0.08",
        )
    return GradeSuggestion(
        style=GradeStyle.DAY_NATURAL,
        reason="daylight footage: natural boost to contrast/saturation, no color-temp shift",
        filter_chain="eq=contrast=1.05:saturation=1.12",
    )
