# AI Video Editor

[![CI](https://github.com/Parakh20/AI-Video-Editor/actions/workflows/ci.yml/badge.svg)](https://github.com/Parakh20/AI-Video-Editor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An AI-powered production pipeline that turns a folder of raw cycling POV
footage into a cinematically edited video: scene/highlight detection, beat
synchronized cuts, subtitles, GPS overlays, chapter markers, an
[OpenTimelineIO](https://opentimelineio.readthedocs.io/) timeline, a Kdenlive
project, and a final rendered MP4 — driven by one command:

```bash
video-editor full ./assets/raw
```

## Quickstart

Requires **Python 3.12** and **ffmpeg** on `PATH`. An NVIDIA GPU is optional:
rendering uses NVENC when the ffmpeg build has it and falls back to libx264
otherwise. See `docs/installation.md` for details.

**Option A: uv, CUDA 12.4 build of PyTorch** (several GB of downloads)

```bash
./scripts/setup.sh                  # installs uv if missing, runs `uv sync --dev`
./scripts/run.sh --help             # or: source .venv/bin/activate && video-editor --help
```

**Option B: pip, CPU-only PyTorch** (what CI uses)

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "setuptools<81" wheel
.venv/bin/pip install --no-build-isolation openai-whisper==20240930
.venv/bin/pip install -e . --group dev
.venv/bin/video-editor --help
```

**Run it.** Raw footage is not part of the repo. Put your clips in
`assets/raw/` (and an optional song in `assets/music/`), or generate a
12-second synthetic test clip:

```bash
ffmpeg -f lavfi -i testsrc2=duration=12:size=1280x720:rate=30 \
       -f lavfi -i sine=frequency=440:duration=12 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest assets/raw/sample.mp4

video-editor full assets/raw --skip-subtitles      # writes everything to output/
video-editor full assets/raw --limit 2 --dry-run   # plan every stage without encoding
video-editor full assets/raw --music assets/music/your_song.mp3
```

`full` writes `scores.json`, `edit_plan.json`, `timeline.otio`,
`project.kdenlive` and `final_render_<preset>.mp4` to `output/`. Subtitles
use Whisper (`base` by default, set in `configs/default.yaml`); the model is
downloaded on first use.

## Commands

| Command | What it does |
|---|---|
| `video-editor analyse <dir>` | Vision (shot detection, highlights) + audio (silence/speech) analysis, cached per clip. |
| `video-editor score <dir>` | Weighted scoring of every shot → `scores.json`, with sub-scores and which signals were available. |
| `video-editor subtitles <path>` | Whisper transcription → `.srt` per clip (gracefully empty on non-speech or audio-less clips). |
| `video-editor timeline` | Edit-decision engine (`scores.json` [+ beat grid, + GPS] → `edit_plan.json`, every decision explained) → `timeline.otio` + `project.kdenlive`. |
| `video-editor render` | Renders the kept, explained edit plan to a final MP4 at a chosen preset. |
| `video-editor full <dir>` | The whole pipeline above, end to end. |

Every command supports `--limit N` (process only the N smallest clips) and
most support `--max-clip-seconds` (analyze only a leading window per clip) —
both are permanent features for fast iteration on large footage libraries,
not just for testing.

## Pipeline

```
raw clips ──► analyse (vision + audio) ──► score (scores.json)
                                                │
              optional GPS/GPX,FIT ────────────┤
              optional song (beat grid) ───────┤
                                                ▼
                                   decision engine (edit_plan.json, explained)
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                                   ▼
                     timeline.otio                        project.kdenlive
                                                │
                                                ▼
                                   render (effects + overlays, NVENC)
                                                │
                                                ▼
                                        final rendered MP4
```

See `docs/architecture.md` for the full module map and design decisions.

## Features by phase

- **Vision**: scene detection, highlight scoring, blur/sharpness, camera
  shake, optical flow, brightness, day/night classification, heuristic scene
  classification, aesthetic scoring, duplicate-frame detection.
- **Audio**: beat/tempo detection, speech detection, Whisper subtitles,
  music energy-section analysis, silence detection.
- **GPS**: GPX/FIT parsing, speed/cadence/HR/power/elevation/gradient/
  distance, climb/descent/high-speed/stop/turn event detection, overlays
  (all gracefully no-op when no GPS file is provided).
- **Scoring & decisions**: a weighted, explainable scoring engine that
  renormalizes over whichever signals are actually present, feeding a
  decision engine that drops/merges/trims/aligns-to-beat and explains every
  choice.
- **Timeline & editors**: editor-agnostic OTIO timeline plus a hand-built
  Kdenlive (MLT XML) project; designed so DaVinci Resolve Studio or Adobe
  Premiere export can be added later against the same `EditAction`/OTIO
  representation.
- **Effects & overlays**: speed ramps, crossfades, fades, zoom/crop,
  two-pass stabilization, letterboxing, LUT application, heuristic
  day/night grading suggestions, cycling telemetry overlays (drawtext +
  matplotlib/PIL, no ImageMagick dependency).
- **Rendering**: NVENC-first (h264/hevc/av1) with automatic libx264
  fallback, presets for 1080p/1440p/4K/YouTube/Instagram Reel/YouTube
  Shorts.

## Project structure

```
src/ai_video_editor/
  cli/          Typer CLI (video-editor), one module per command, config loading
  vision/       shot detection, highlights, blur/sharpness, shake, flow, brightness, scene/day-night
  audio/        beats/tempo, speech and silence detection, Whisper subtitles, music sections
  gps/          GPX/FIT parsing, ride metrics, event detection, telemetry overlays
  analysis/     weighted, explainable shot scoring
  editing/      decision engine (keep/drop/merge/trim/beat-align) and effects
  timeline/     OpenTimelineIO builder and Kdenlive (MLT XML) export
  render/       ffmpeg renderer and output presets
  utils/        logging and video I/O helpers
configs/default.yaml   scoring weights, decision thresholds, model sizes
scripts/        setup/run/update wrappers and manual smoke scripts for real footage
tests/          pytest suite mirroring the package layout, plus a synthetic GPX fixture
docs/           architecture, developer guide, installation, editor integration, report
assets/raw, assets/music   your footage and music (gitignored)
```

Configuration lives in `configs/default.yaml`; there are no environment
variables or API keys.

## Known limitations

- **Kdenlive projects are checked structurally, not in the GUI.**
  `project.kdenlive` is validated as well-formed MLT XML with the expected
  structure. See `docs/editor_integration.md` for the verification command.
- **DaVinci Resolve and Premiere export are not implemented.** The
  `EditAction`/OTIO representation is designed so they can be added later.
- **GPS features are tested on a synthetic ride.** The original test footage
  had no GPX/FIT recording, so GPS scoring and overlays are exercised against
  `tests/fixtures/sample_ride.gpx`.
- **Whisper transcribes wind/engine noise into low-value text on
  non-speech action-cam audio.** That is how the model behaves on this kind
  of footage.
- Full-resolution analysis of many multi-gigabyte 4K clips is slow; use
  `--limit`/`--max-clip-seconds` while iterating.

## Development

```bash
.venv/bin/python -m pytest tests/ --cov=ai_video_editor --cov-report=term-missing
.venv/bin/python -m ruff check src/
.venv/bin/python -m mypy src/
```

Tests build synthetic video, audio and GPX fixtures, so they need ffmpeg but
no footage, GPU or network. Tests that need inputs which are not in the repo
skip with a reason: the CLI smoke tests run only when real clips are in
`assets/raw/`, and the Whisper tests run only when the `tiny` model is
already cached (`python -c "import whisper; whisper.load_model('tiny')"`).
CI runs ruff and the test suite on every push and pull request.

See `docs/developer_guide.md` for how to add a new analysis signal, render
preset, or editor export target, and `docs/final_report.md` for the full
project report (folder tree, architecture/dependency diagrams, benchmarks,
future work).

## License

MIT. See [LICENSE](LICENSE).
