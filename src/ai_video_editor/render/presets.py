"""Typed render presets/profiles for output resolution and platform targets (Phase 12)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AspectMode(str, Enum):
    """How to fit source footage into a target aspect ratio."""

    NATIVE = "native"
    """No crop/pad — output aspect matches the source."""
    CENTER_CROP = "center_crop"
    """Crop the longer axis to hit the target aspect (e.g. 16:9 -> 9:16)."""
    LETTERBOX = "letterbox"
    """Pad with black bars to hit the target aspect without cropping."""


@dataclass(frozen=True)
class RenderPreset:
    name: str
    width: int
    height: int
    fps: float
    video_bitrate: str
    audio_bitrate: str
    aspect_mode: AspectMode


PRESETS: dict[str, RenderPreset] = {
    "1080p": RenderPreset(
        name="1080p",
        width=1920,
        height=1080,
        fps=29.97,
        video_bitrate="12M",
        audio_bitrate="192k",
        aspect_mode=AspectMode.NATIVE,
    ),
    "1440p": RenderPreset(
        name="1440p",
        width=2560,
        height=1440,
        fps=29.97,
        video_bitrate="20M",
        audio_bitrate="192k",
        aspect_mode=AspectMode.NATIVE,
    ),
    "4k": RenderPreset(
        name="4k",
        width=3840,
        height=2160,
        fps=29.97,
        video_bitrate="45M",
        audio_bitrate="192k",
        aspect_mode=AspectMode.NATIVE,
    ),
    "youtube": RenderPreset(
        name="youtube",
        width=1920,
        height=1080,
        fps=29.97,
        video_bitrate="12M",
        audio_bitrate="192k",
        aspect_mode=AspectMode.LETTERBOX,
    ),
    "instagram_reel": RenderPreset(
        name="instagram_reel",
        width=1080,
        height=1920,
        fps=30.0,
        video_bitrate="8M",
        audio_bitrate="128k",
        aspect_mode=AspectMode.CENTER_CROP,
    ),
    "youtube_shorts": RenderPreset(
        name="youtube_shorts",
        width=1080,
        height=1920,
        fps=30.0,
        video_bitrate="10M",
        audio_bitrate="128k",
        aspect_mode=AspectMode.CENTER_CROP,
    ),
}


def get_preset(name: str) -> RenderPreset:
    """Look up a preset by name (case-insensitive), raising ``KeyError`` with
    the list of valid names if unknown."""
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown render preset {name!r}; valid presets: {sorted(PRESETS)}")
    return PRESETS[key]
