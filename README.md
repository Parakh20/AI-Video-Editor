# AI Video Editor

An AI-powered production pipeline that turns a folder of raw cycling POV
footage into a cinematically edited video: scene/highlight detection, beat
synchronized cuts, subtitles, GPS overlays, chapter markers, an
[OpenTimelineIO](https://opentimelineio.readthedocs.io/) timeline, a Kdenlive
project, and a final rendered MP4 — driven by one command:

```bash
video-editor full ./assets/raw
```

## Quickstart

```bash
./scripts/setup.sh                  # installs uv (if missing) and syncs the venv
./scripts/run.sh --help              # or: source .venv/bin/activate && video-editor --help
./scripts/run.sh full assets/raw --limit 2 --dry-run   # plan without encoding
./scripts/run.sh full assets/raw --music assets/music/your_song.mp3
```

Requires **Python 3.12**, **ffmpeg** (with `h264_nvenc`/`hevc_nvenc` +
`vidstab` for GPU encoding and stabilization — optional but recommended), and
an **NVIDIA GPU + driver** for hardware-accelerated encoding/inference
(falls back to CPU automatically otherwise). See `docs/installation.md`.

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

## Known limitations

- **Kdenlive is not installed in this development environment** (`apt
  install kdenlive` needs sudo, not automatable from here) — `project.kdenlive`
  is validated as well-formed MLT XML with the expected structure, but has
  not been opened in the Kdenlive GUI. See `docs/editor_integration.md` for
  the exact verification command.
- **DaVinci Resolve Studio integration is designed for, but not active.**
  A `davinci-resolve-mcp` server is already installed and configured on this
  machine, but only the free edition of Resolve is licensed here, so its
  scripting API is unavailable until Resolve Studio is activated.
- **No GPS telemetry exists for the real test footage** (`assets/raw`, a
  real cycling ride with no GPX/FIT recording) — GPS overlay/scoring code
  is fully implemented and tested against a synthetic fixture
  (`tests/fixtures/sample_ride.gpx`), but has no real ride data to run
  against in this project as-is.
- **Whisper transcribes wind/engine noise into low-value text on
  non-speech action-cam audio** by design of the underlying model — this is
  expected behavior on this kind of footage, not a bug.
- Full-resolution analysis of many multi-gigabyte 4K clips is
  compute/time-intensive; use `--limit`/`--max-clip-seconds` for iteration.

## Development

```bash
.venv/bin/python -m pytest tests/ --cov=ai_video_editor --cov-report=term-missing
.venv/bin/python -m ruff check src/
.venv/bin/python -m mypy src/
```

See `docs/developer_guide.md` for how to add a new analysis signal, render
preset, or editor export target, and `docs/final_report.md` for the full
project report (folder tree, architecture/dependency diagrams, benchmarks,
future work).
