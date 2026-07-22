# Architecture

## Module map

```
src/ai_video_editor/
├── vision/       shot/highlight detection, per-frame quality signals (no cross-package deps)
├── audio/        beat/tempo/speech/subtitle/music/silence analysis (no cross-package deps)
├── gps/          GPX/FIT parsing, metrics, event detection, overlays (no cross-package deps)
├── analysis/     scoring — imports vision, audio, gps
├── editing/      decision engine + effects — imports analysis, audio (beat grid)
├── timeline/     OTIO + Kdenlive export — imports editing (EditAction)
├── render/       ffmpeg NVENC renderer + presets — imports editing (effects), timeline
├── cli/          `video-editor` command surface — imports everything above
└── utils/        shared logging, video I/O — imported by everyone, imports nothing internal
```

`vision`, `audio`, and `gps` are intentionally siblings with zero
cross-imports between them — each is independently testable and could be
used standalone. Everything downstream composes them.

## Data flow

```
raw clips (assets/raw)
   │
   ▼
analyse ─── vision.scene_detection + per-frame signals
        └── audio.silence_detection / speech_detection
   │  (cached per clip under cache/)
   ▼
score ─── analysis.scoring.score_video()
   │  → scores.json  (per-shot score + sub-scores + which signals were available)
   ▼
timeline ─── editing.decision_engine.build_edit_plan()
   │        (scores.json + optional beat grid from audio.beat_detection
   │         + optional gps.event_detection)
   │  → edit_plan.json  (ordered EditActions, every one with an `explanation`)
   │
   ├──► timeline.otio_builder.build_otio_timeline()   → timeline.otio
   └──► timeline.kdenlive_export.build_kdenlive_project() → project.kdenlive
   │
   ▼
render ─── render.renderer (reads edit_plan.json directly, applies
   │        editing.effects + gps.overlays, ffmpeg filter_complex, NVENC)
   ▼
final rendered MP4 (output/)
```

`subtitles` (Whisper, via `audio.subtitle_generation`) runs independently
per clip and is not on the scoring/render critical path — it produces
`.srt` files alongside the rest of the pipeline.

## Key design decisions

- **Scoring renormalizes over available signals.** `analysis.scoring`'s
  weights are only applied to signals that actually exist for a given shot
  (e.g. GPS-derived sub-scores are entirely excluded, not zero-filled, when
  no GPS file is provided) so a signal's absence never silently deflates a
  score. This mattered immediately: the real project footage has no GPS
  data at all.
- **Motion is scaled by stability, not scored independently.** A
  high-motion-but-shaky shot scores lower than a high-motion-but-stable
  one — the design goal is "energetic but controlled" cycling footage, not
  "movement is always good."
- **`filter_complex` over the concat demuxer for rendering.** Kept segments
  need arbitrary per-clip trims and mixed transition types composed in one
  pass; the concat demuxer can't express that without disk-heavy
  intermediate re-encodes per segment.
- **ffmpeg `drawtext`/`overlay`, not MoviePy `TextClip`, for all text and
  overlays.** `TextClip` requires ImageMagick's `convert`/`magick` binary,
  which is not installed in this environment and can't be installed without
  sudo. Overlays that need image composition (elevation graph, route map,
  stats card) go through `matplotlib`/`PIL` instead, both pure-Python.
- **Kdenlive MLT XML is hand-built, not converted from OTIO**, because no
  MLT/Kdenlive adapter is registered in this project's OpenTimelineIO
  install (`otio.adapters.available_adapter_names()` only lists
  `otio_json`/`otioz`/`otiod`). `timeline/kdenlive_export.py` builds the
  `<mlt>`/`<producer>`/`<playlist>`/`<tractor>` structure directly against
  Kdenlive's documented project format.
- **NVENC-first, libx264/libx265 fallback.** The renderer always attempts
  hardware encoding first and falls back to a CPU encoder automatically if
  NVENC is unavailable or fails, rather than requiring the caller to know
  which is available.
- **Editor-agnostic by construction.** Nothing above `timeline/` knows
  about a specific editor; adding DaVinci Resolve Studio or Adobe Premiere
  export later means adding a new module beside `kdenlive_export.py` that
  consumes the same `list[EditAction]`/OTIO timeline, not touching scoring
  or decision logic. See `docs/developer_guide.md`.
