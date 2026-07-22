"""YAML-backed configuration for the CLI.

Loads `configs/default.yaml` (or a path given by the user) into a typed
:class:`PipelineConfig`. CLI flags always take precedence over config file
values -- commands read the config first, then apply any explicit flag the
user passed, never the other way around.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ai_video_editor.analysis.scoring import ScoreWeights
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class DecisionConfig(BaseModel):
    score_threshold: float = 30.0
    merge_gap_tolerance: float = 1.0
    beat_tolerance: float = 0.15
    trim_seconds: float = 0.15


class SceneDetectionConfig(BaseModel):
    threshold: float = 27.0
    min_shot_duration: float = 0.5


class SubtitlesConfig(BaseModel):
    whisper_model_size: str = "base"


class RenderConfig(BaseModel):
    default_preset: str = "1080p"
    fps: float = 29.97
    crossfade_seconds: float = 0.5


class PipelineConfig(BaseModel):
    scoring: ScoreWeights = Field(default_factory=ScoreWeights)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    scene_detection: SceneDetectionConfig = Field(default_factory=SceneDetectionConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load the pipeline config from YAML, falling back to defaults for any
    section/key that is missing or if the file doesn't exist at all."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file %s not found; using built-in defaults.", config_path)
        return PipelineConfig()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    logger.info("Loaded config from %s", config_path)
    return PipelineConfig.model_validate(raw)
