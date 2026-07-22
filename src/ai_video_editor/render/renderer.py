"""FFmpeg-driven rendering of a kept edit plan to a final output file (Phase 12).

Uses ``-filter_complex`` rather than the concat demuxer: kept segments need
per-clip trims (arbitrary in/out points into large source files) and mixed
transition types (hard cut vs. crossfade) between adjacent clips, both of
which the concat demuxer can't express in a single pass. Building one
filter graph also avoids writing intermediate re-encoded segment files to
disk, which matters here given limited free space on large source footage.

Encoding prefers NVENC (GPU) and falls back to libx264/libx265 (CPU) if
NVENC is unavailable or the encode fails for any reason (e.g. no NVIDIA
driver, VRAM exhausted).

Real DJI action-cam source footage is not uniform: some clips have an AAC
audio stream, some have none at all (confirmed against real project
footage — a short 43MB clip has zero audio streams while its siblings all
have one). Every clip is probed for an audio stream before the filter
graph is built; clips without one get a synthesized silent ``anullsrc``
input so the audio side of the graph stays well-formed regardless of what
the source footage actually contains.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ai_video_editor.editing.decision_engine import EditAction
from ai_video_editor.gps.overlays import MarkerWindow
from ai_video_editor.render.presets import AspectMode, RenderPreset
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CROSSFADE_S = 0.5


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for safe interpolation into an ffmpeg filtergraph.

    ffmpeg filtergraph syntax treats `:`, `'`, `\`, and other characters as
    structural tokens within filter option values. This function escapes them
    to prevent the path from corrupting the filter graph string.
    """
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    used_encoder: str
    clip_count: int


def _has_audio_stream(video_path: str, *, ffprobe_bin: str = "ffprobe") -> bool:
    """Return whether ``video_path`` has at least one audio stream."""
    result = subprocess.run(
        [
            ffprobe_bin, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


@dataclass(frozen=True)
class _ClipInputs:
    """Input-stream indices for one kept clip, after audio-presence probing."""

    video_input_index: int
    audio_input_index: int
    audio_synthesized: bool


@dataclass(frozen=True)
class MarkerOverlaySpec:
    """A moving-dot marker composited over a :class:`GraphicOverlaySpec`."""

    image_path: Path
    windows: list[MarkerWindow]
    """Positions are relative to the parent GraphicOverlaySpec's own (x, y);
    the caller is responsible for centering (e.g. subtracting the dot's own
    half-width/half-height) before constructing this spec."""


@dataclass(frozen=True)
class GraphicOverlaySpec:
    """A static background image (e.g. elevation profile) plus an optional
    moving marker, composited into the render at a fixed screen position."""

    image_path: Path
    x: int
    y: int
    marker: MarkerOverlaySpec | None = None


def _build_inputs(kept: list[EditAction]) -> tuple[list[str], list[_ClipInputs]]:
    """Build the ffmpeg ``-i``/input arg list, probing each clip for audio.

    Clips with no audio stream get an additional ``anullsrc`` lavfi input
    (silent, matching the clip's kept duration) rather than being mapped to
    a non-existent ``[i:a]`` stream, which ffmpeg rejects outright.
    """
    input_args: list[str] = []
    clip_inputs: list[_ClipInputs] = []
    next_index = 0

    for clip in kept:
        input_args += [
            "-ss", f"{clip.start_seconds:.3f}",
            "-t", f"{clip.duration_seconds:.3f}",
            "-i", clip.video_path,
        ]
        video_index = next_index
        next_index += 1

        if _has_audio_stream(clip.video_path):
            clip_inputs.append(
                _ClipInputs(video_input_index=video_index, audio_input_index=video_index, audio_synthesized=False)
            )
            continue

        logger.info(
            "render_edit_plan: %s has no audio stream, synthesizing silence for its %.2fs segment",
            clip.video_path,
            clip.duration_seconds,
        )
        input_args += [
            "-f", "lavfi",
            "-t", f"{clip.duration_seconds:.3f}",
            "-i", "anullsrc=r=48000:cl=stereo",
        ]
        audio_index = next_index
        next_index += 1
        clip_inputs.append(
            _ClipInputs(video_input_index=video_index, audio_input_index=audio_index, audio_synthesized=True)
        )

    return input_args, clip_inputs


def _scale_and_fit_filter(preset: RenderPreset) -> str:
    """Return the scale(+crop/pad) fragment that fits a clip into the preset frame."""
    w, h = preset.width, preset.height
    if preset.aspect_mode is AspectMode.CENTER_CROP:
        # Scale to cover the target box, then center-crop the overflow.
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1"
        )
    # NATIVE and LETTERBOX both fit-within (no cropping) then pad to the
    # exact target box. NATIVE is "no crop", not "no pad": a rendered
    # output has one fixed resolution for its whole duration, and source
    # clips are not guaranteed to share an aspect ratio (confirmed against
    # real project footage — sibling DJI clips differ in frame size), so
    # every clip must be normalized to identical WxH or xfade/concat reject
    # the mismatched frame sizes outright.
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )


def _build_graphic_overlay_fragments(
    acc_v: str,
    graphic_overlays: Sequence[GraphicOverlaySpec],
) -> tuple[list[str], str]:
    """Build filter fragments compositing background images + moving markers.

    Each background image loads via ``movie=`` (no extra ffmpeg ``-i``
    needed) and composites once at a fixed screen position. An optional
    marker dot is split into one copy per time window and chained as
    individually time-gated ``overlay`` filters, mirroring how the text
    overlay chains gated ``drawtext`` filters.
    """
    parts: list[str] = []
    for i, spec in enumerate(graphic_overlays):
        bg_label = f"bg{i}"
        parts.append(f"movie={_escape_filter_path(spec.image_path)}[{bg_label}]")
        next_v = f"gv{i}"
        parts.append(f"[{acc_v}][{bg_label}]overlay={spec.x}:{spec.y}[{next_v}]")
        acc_v = next_v

        if spec.marker is None or not spec.marker.windows:
            continue

        windows = spec.marker.windows
        dot_src = f"dotsrc{i}"
        parts.append(f"movie={_escape_filter_path(spec.marker.image_path)}[{dot_src}]")
        dot_labels = [f"dot{i}_{j}" for j in range(len(windows))]
        if len(dot_labels) == 1:
            parts.append(f"[{dot_src}]copy[{dot_labels[0]}]")
        else:
            fanout = "".join(f"[{label}]" for label in dot_labels)
            parts.append(f"[{dot_src}]split={len(dot_labels)}{fanout}")

        for j, (window, dot_label) in enumerate(zip(windows, dot_labels, strict=True)):
            marker_out = f"mk{i}_{j}"
            x = spec.x + window.x_px
            y = spec.y + window.y_px
            parts.append(
                f"[{acc_v}][{dot_label}]overlay={x:.1f}:{y:.1f}:"
                f"enable='between(t,{window.start_s:.3f},{window.end_s:.3f})'[{marker_out}]"
            )
            acc_v = marker_out

    return parts, acc_v


def _build_filter_complex(
    kept: list[EditAction],
    clip_inputs: list[_ClipInputs],
    preset: RenderPreset,
    *,
    crossfade_s: float,
    overlay_drawtext: str | None,
    graphic_overlays: Sequence[GraphicOverlaySpec] = (),
) -> tuple[str, str, str]:
    """Build the filter_complex graph string plus final video/audio labels."""
    fit = _scale_and_fit_filter(preset)
    parts: list[str] = []

    for i, ci in enumerate(clip_inputs):
        parts.append(f"[{ci.video_input_index}:v]{fit},fps={preset.fps}[v{i}]")
        parts.append(f"[{ci.audio_input_index}:a]aresample=48000[a{i}]")

    acc_v, acc_a = "v0", "a0"
    cumulative_duration = kept[0].duration_seconds

    for i in range(1, len(kept)):
        clip = kept[i]
        v_in, a_in = f"v{i}", f"a{i}"
        if clip.transition_in == "crossfade" and cumulative_duration > crossfade_s:
            offset = cumulative_duration - crossfade_s
            v_out, a_out = f"vx{i}", f"ax{i}"
            parts.append(
                f"[{acc_v}][{v_in}]xfade=transition=fade:"
                f"duration={crossfade_s:.3f}:offset={offset:.3f}[{v_out}]"
            )
            parts.append(
                f"[{acc_a}][{a_in}]acrossfade=d={crossfade_s:.3f}[{a_out}]"
            )
            cumulative_duration += clip.duration_seconds - crossfade_s
        else:
            v_out, a_out = f"vc{i}", f"ac{i}"
            parts.append(f"[{acc_v}][{v_in}]concat=n=2:v=1:a=0[{v_out}]")
            parts.append(f"[{acc_a}][{a_in}]concat=n=2:v=0:a=1[{a_out}]")
            cumulative_duration += clip.duration_seconds
        acc_v, acc_a = v_out, a_out

    if overlay_drawtext:
        final_v = "vout"
        parts.append(f"[{acc_v}]{overlay_drawtext}[{final_v}]")
        acc_v = final_v

    graphic_parts, acc_v = _build_graphic_overlay_fragments(acc_v, graphic_overlays)
    parts += graphic_parts

    return ";".join(parts), acc_v, acc_a


def _encoder_args(codec: str, preset: RenderPreset) -> list[str]:
    if codec == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-b:v", preset.video_bitrate,
        ]
    return [
        "-c:v", "libx264",
        "-preset", "medium",
        "-b:v", preset.video_bitrate,
    ]


def render_edit_plan(
    edit_actions: list[EditAction],
    output_path: Path,
    preset: RenderPreset,
    *,
    crossfade_s: float = _DEFAULT_CROSSFADE_S,
    overlay_drawtext: str | None = None,
    graphic_overlays: Sequence[GraphicOverlaySpec] = (),
    prefer_gpu: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> RenderResult:
    """Render the kept segments of an edit plan to a single output file.

    ``overlay_drawtext`` is an optional pre-built ``drawtext=...`` filter
    fragment (see :mod:`ai_video_editor.gps.overlays`) burned in after all
    transitions are composed.
    """
    kept = [a for a in edit_actions if a.kind == "keep"]
    if not kept:
        raise ValueError("render_edit_plan: edit plan has no kept segments to render")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_args, clip_inputs = _build_inputs(kept)
    filter_complex, v_label, a_label = _build_filter_complex(
        kept,
        clip_inputs,
        preset,
        crossfade_s=crossfade_s,
        overlay_drawtext=overlay_drawtext,
        graphic_overlays=graphic_overlays,
    )

    encoders_to_try = ["h264_nvenc", "libx264"] if prefer_gpu else ["libx264"]
    if not shutil.which(ffmpeg_bin):
        raise RuntimeError(f"render_edit_plan: ffmpeg binary {ffmpeg_bin!r} not found on PATH")

    last_error: subprocess.CalledProcessError | None = None
    for codec in encoders_to_try:
        cmd = (
            [ffmpeg_bin, "-y"]
            + input_args
            + [
                "-filter_complex", filter_complex,
                "-map", f"[{v_label}]",
                "-map", f"[{a_label}]",
            ]
            + _encoder_args(codec, preset)
            + ["-c:a", "aac", "-b:a", preset.audio_bitrate, "-r", str(preset.fps)]
            + [str(output_path)]
        )
        logger.info("render_edit_plan: encoding with %s (%d clips)", codec, len(kept))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("render_edit_plan: wrote %s using %s", output_path, codec)
            return RenderResult(output_path=output_path, used_encoder=codec, clip_count=len(kept))
        logger.warning(
            "render_edit_plan: encoder %s failed (rc=%d), trying next option.\nstderr tail: %s",
            codec,
            result.returncode,
            "\n".join(result.stderr.strip().splitlines()[-15:]),
        )
        last_error = subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    assert last_error is not None
    raise last_error
