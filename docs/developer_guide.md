# Developer guide

## Adding a new analysis signal

1. Add a module to `src/ai_video_editor/vision/`, `audio/`, or `gps/`
   following the existing pattern: full type hints, a typed
   dataclass/pydantic return type, `get_logger(__name__)` from
   `utils/logging.py`, no cross-imports between `vision`/`audio`/`gps`
   themselves.
2. Wire it into `analysis/scoring.py`'s `ScoreWeights`/`SubScores`/
   `score_shot()` — add a new weighted field, and make sure it's excluded
   (not zero-filled) from the weighted sum when its input signal is
   unavailable, matching the existing GPS/audio/music-sync handling.
3. Add unit tests against synthetic fixtures (see `tests/vision/conftest.py`
   / `tests/audio/conftest.py` / `tests/gps/` for the existing patterns —
   generate tiny synthetic frames/audio/GPX on the fly, don't depend on
   real footage for fast tests).
4. If it's expensive enough to need real-footage verification, add one
   `@pytest.mark.slow` test against the smallest clip in `assets/raw/`
   (see `tests/cli/conftest.py`'s `smallest_real_clip` fixture).

## Adding a new render preset

Add an entry to `render/presets.py` (resolution, bitrate, fps,
aspect-ratio/crop behavior) and a corresponding case in
`render/renderer.py` if the preset needs special handling (e.g. a
9:16 crop for Reels/Shorts). Add a preset-specific test in
`tests/render/`.

## Adding a new editor export target

The pipeline is deliberately editor-agnostic above `timeline/`: everything
downstream of `editing/decision_engine.py`'s `list[EditAction]` (and the
equivalent OTIO `Timeline` from `timeline/otio_builder.py`) is a pure
export step. To add e.g. Adobe Premiere (XML/AAF) or DaVinci Resolve
Studio:

1. Add a new module beside `timeline/kdenlive_export.py`, e.g.
   `timeline/premiere_export.py` or `timeline/resolve_export.py`, that
   consumes the same `list[EditAction]` or `otio.schema.Timeline` — do not
   modify `analysis/`, `editing/decision_engine.py`, or `editing/effects.py`.
2. Add a corresponding `video-editor timeline --export <target>` option
   (or a new command) in `cli/commands/timeline.py`.
3. **DaVinci Resolve Studio specifically**: a `davinci-resolve-mcp` server
   is already installed and configured on this machine at
   `~/.local/share/davinci-resolve-mcp/` (see its own `AGENTS.md`/`README.md`
   for its extensive tool surface — media pool, timeline, color, render,
   etc.). It is currently unusable because only the free edition of
   Resolve is licensed on this machine (confirmed: no `External scripting
   using` preference exists in this build's UI, `fusionscript.so` loads
   but `scriptapp("Resolve")` returns `None`, and `/opt/resolve/.license/`
   is empty). Once Resolve Studio is licensed and activated, that MCP
   server becomes the natural integration point — either drive it directly
   from a new `timeline/resolve_export.py` that imports the edit plan via
   the MCP server's project/timeline tools, or export OTIO and use
   Resolve's own OTIO import.

## Testing conventions

- **Fast unit tests** (the default `pytest` run) use synthetic fixtures
  generated at test time — synthetic video frames via OpenCV, synthetic
  audio via numpy/soundfile, a synthetic GPX track
  (`tests/fixtures/sample_ride.gpx`). These must not depend on real
  footage or take more than a second or two each.
- **`@pytest.mark.slow`** tests exercise real ffmpeg encodes, real Whisper
  transcription, or real clips from `assets/raw/` (always via the smallest
  clip(s), and/or `--max-clip-seconds`, to keep runtime bounded). Run the
  full suite including these with `pytest tests/`; skip them for fast
  iteration with `pytest tests/ -m "not slow"`.
- Every module's public functions are fully typed; `mypy src/` and
  `ruff check src/` must both be clean before considering a change done.

## Config

Tunable defaults (score weights, decision thresholds, Whisper model size,
render preset/fps) live in `configs/default.yaml`, loaded via
`cli/config.py`. CLI flags always override config file values.
