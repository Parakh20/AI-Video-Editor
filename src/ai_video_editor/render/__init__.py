"""Render presets and ffmpeg-driven output rendering (Phase 12)."""

from ai_video_editor.render.presets import PRESETS, AspectMode, RenderPreset, get_preset
from ai_video_editor.render.renderer import RenderResult, render_edit_plan

__all__ = [
    "PRESETS",
    "AspectMode",
    "RenderPreset",
    "RenderResult",
    "get_preset",
    "render_edit_plan",
]
