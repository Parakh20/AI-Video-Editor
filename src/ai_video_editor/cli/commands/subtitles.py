"""`video-editor subtitles` -- generate SRT subtitles for a clip or directory via Whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from ai_video_editor.audio.subtitle_generation import SubtitleResult, generate_subtitles
from ai_video_editor.cli.config import load_config
from ai_video_editor.cli.main import app
from ai_video_editor.cli.pipeline import VIDEO_EXTENSIONS
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "subtitles"


def _targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix in VIDEO_EXTENSIONS and p.is_file())


def generate_subtitles_for(
    path: Path, *, output_dir: Path, model_size: str, limit: int | None = None
) -> dict[str, SubtitleResult]:
    """Generate SRT subtitles for a single file or every clip in a directory.

    Shared by the `subtitles` and `full` commands. Returns a mapping of clip
    stem -> :class:`SubtitleResult`; clips with no detected speech still get
    an entry (with an empty transcript) rather than being silently skipped.
    """
    targets = _targets(path)
    if limit is not None:
        targets = targets[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, SubtitleResult] = {}
    for clip in track(targets, description="Transcribing..."):
        try:
            result = generate_subtitles(clip, model_size=model_size)
        except Exception as exc:  # noqa: BLE001 - a clip with no/corrupt audio must not abort the batch
            logger.warning("subtitles(%s): transcription failed (%s)", clip.name, exc)
            continue
        results[clip.stem] = result
        srt_path = output_dir / f"{clip.stem}.srt"
        srt_path.write_text(result.srt_text, encoding="utf-8")
        console.print(f"  [cyan]{clip.name}[/cyan]: {len(result.segments)} segment(s) -> {srt_path.name}")
    return results


@app.command()
def subtitles(
    path: Annotated[
        Path, typer.Argument(help="A single video file, or a directory of clips.")
    ] = PROJECT_ROOT / "assets" / "raw",
    model_size: Annotated[
        str | None, typer.Option(help="Whisper model size (tiny/base/small/medium/large).")
    ] = None,
    output_dir: Annotated[Path, typer.Option(help="Where to write .srt files.")] = DEFAULT_OUTPUT_DIR,
    limit: Annotated[int | None, typer.Option(help="Only transcribe the first N clips.")] = None,
) -> None:
    """Transcribe speech into timed .srt subtitle files via Whisper.

    Cycling POV footage often has little or no speech; clips with no
    detected speech still produce an (empty) subtitle entry rather than an
    error -- this is expected Whisper behaviour, not a failure.
    """
    config = load_config()
    resolved_model = model_size or config.subtitles.whisper_model_size
    results = generate_subtitles_for(path, output_dir=output_dir, model_size=resolved_model, limit=limit)
    total_segments = sum(len(r.segments) for r in results.values())
    console.print(
        f"[bold green]Transcribed {len(results)} clip(s)[/bold green], "
        f"{total_segments} subtitle segment(s) total -> {output_dir}"
    )
