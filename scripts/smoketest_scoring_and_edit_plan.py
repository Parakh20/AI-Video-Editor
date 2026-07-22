"""One-off integration smoke test: score real Kolar clips + build a real edit plan.

Not a pytest test (too slow/heavy for the regular suite) — run manually:
    .venv/bin/python scripts/smoketest_scoring_and_edit_plan.py
"""

from __future__ import annotations

from pathlib import Path

from ai_video_editor.analysis.scoring import score_video, write_scores_json
from ai_video_editor.audio.beat_detection import detect_beats
from ai_video_editor.editing.decision_engine import build_edit_plan, write_edit_plan_json
from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "assets" / "raw"
SONG_PATH = PROJECT_ROOT / "assets" / "music" / "Kesha - TiK ToK (Lyrics) - 7clouds (youtube).mp3"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Smallest three real clips by file size, for a fast but real smoke test.
CLIP_NAMES = [
    "DJI_20260712185038_0133_D.MP4",
    "DJI_20260712213106_0141_D.MP4",
    "DJI_20260712191345_0088_D.MP4",
]
CLIP_MAX_SECONDS = 15.0  # only analyze a short prefix of each clip


def main() -> None:
    logger.info("Detecting beat grid from real soundtrack: %s", SONG_PATH)
    beat_grid = detect_beats(SONG_PATH)
    logger.info("Beat grid: %.1f BPM, %d beats over %.1fs", beat_grid.tempo_bpm, beat_grid.beat_count, beat_grid.duration_seconds)

    all_scores = []
    for name in CLIP_NAMES:
        clip_path = RAW_DIR / name
        if not clip_path.exists():
            logger.warning("Clip not found, skipping: %s", clip_path)
            continue
        logger.info("Scoring %s (first %.0fs, no GPS data available for this ride)", name, CLIP_MAX_SECONDS)
        scores = score_video(clip_path, max_seconds=CLIP_MAX_SECONDS)
        logger.info("  -> %d shots scored", len(scores))
        all_scores.extend(scores)

    if not all_scores:
        raise RuntimeError("No clips scored — check assets/raw and CLIP_NAMES")

    write_scores_json(all_scores, OUTPUT_DIR / "scores_smoketest.json")

    edit_plan = build_edit_plan(all_scores, beat_grid=beat_grid)
    write_edit_plan_json(edit_plan, OUTPUT_DIR / "edit_plan_smoketest.json")

    kept = [a for a in edit_plan if a.kind == "keep"]
    dropped = [a for a in edit_plan if a.kind == "drop"]
    print("\n=== Smoke test summary ===")
    print(f"Clips analyzed: {len(CLIP_NAMES)}")
    print(f"Segments scored: {len(all_scores)}")
    print(f"Edit plan actions: {len(edit_plan)} ({len(kept)} keep, {len(dropped)} drop)")
    print(f"GPS available for any segment: {any(s.gps_available for s in all_scores)} (expected False — no GPS for this ride)")
    print("\nSample kept action:")
    if kept:
        print(f"  {kept[0].explanation}")
    print("\nSample dropped action:")
    if dropped:
        print(f"  {dropped[0].explanation}")
    print(f"\nWrote: {OUTPUT_DIR / 'scores_smoketest.json'}")
    print(f"Wrote: {OUTPUT_DIR / 'edit_plan_smoketest.json'}")


if __name__ == "__main__":
    main()
