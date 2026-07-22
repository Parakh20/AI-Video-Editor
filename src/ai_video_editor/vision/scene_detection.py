"""Shot boundary detection via PySceneDetect's content-aware detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONTENT_THRESHOLD = 27.0  # PySceneDetect's own recommended default


@dataclass(frozen=True, slots=True)
class Shot:
    index: int
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def detect_shots(
    video_path: Path,
    *,
    threshold: float = DEFAULT_CONTENT_THRESHOLD,
    max_seconds: float | None = None,
) -> list[Shot]:
    """Detect shot boundaries in a video, returning a list of contiguous Shots.

    If no boundaries are found, the whole (bounded) video is returned as one Shot.
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    duration = max_seconds if max_seconds is not None else None
    scene_manager.detect_scenes(video, duration=duration)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        end = max_seconds if max_seconds is not None else video.duration.get_seconds()
        logger.info("scene_detection(%s): no cuts found, returning single shot", video_path.name)
        return [Shot(index=0, start_seconds=0.0, end_seconds=end)]

    shots = [
        Shot(index=i, start_seconds=start.get_seconds(), end_seconds=end.get_seconds())
        for i, (start, end) in enumerate(scene_list)
    ]
    logger.info("scene_detection(%s): found %d shots", video_path.name, len(shots))
    return shots
