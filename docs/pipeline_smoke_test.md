# Real end-to-end pipeline smoke test

The default `pytest` suite deliberately keeps `full`'s real-data test to a
single tiny clip with `--dry-run` (see `tests/cli/test_full.py`), so the
suite stays fast. The genuine, non-dry-run, multi-clip run -- the project's
actual success criterion -- takes several minutes against real multi-gigabyte
footage and is documented here as a manual/CI-optional smoke test instead.

## Running it

```bash
./scripts/run.sh full assets/raw \
  --music "assets/music/Kesha - TiK ToK (Lyrics) - 7clouds (youtube).mp3" \
  --limit 4
```

`--limit 4` picks the 4 smallest clips (by file size) from `assets/raw`,
which keeps a full run to single-digit minutes instead of the ~20-30
minutes a full 34-clip, multi-gigabyte batch would take. Drop `--limit`
entirely once you're ready to process everything.

## What to check afterwards

- `output/scores.json` -- every scored shot, with sub-score breakdown.
- `output/edit_plan.json` -- every keep/drop decision, each with a human-readable `explanation`.
- `output/timeline.otio` -- editor-agnostic timeline; open with any OTIO-aware tool or re-read via
  `opentimelineio.adapters.read_from_file`.
- `output/project.kdenlive` -- open in Kdenlive once it's installed (`sudo apt install kdenlive`)
  to confirm it imports cleanly; this environment doesn't have Kdenlive installed, so this file has
  only been validated as well-formed MLT XML, not opened in the GUI.
- `output/final_render_<preset>.mp4` -- the rendered output; `ffprobe` it to confirm duration/streams.
- `output/subtitles/*.srt` -- per-clip subtitles (empty for clips with no detected speech, which is
  expected for cycling POV footage).

## Known real-footage quirks already handled

- Some DJI source clips have no audio stream at all; scoring/rendering treat this as
  "audio unavailable" rather than failing (see `cli/pipeline.analyse_clip_audio` and
  `render/renderer._has_audio_stream`).
- Source clips vary in native resolution/aspect ratio; the renderer always pads to the target
  preset's exact pixel dimensions so `xfade`/`concat` never see mismatched frame sizes.
- This ride has no GPX/FIT telemetry; every GPS-dependent signal and overlay is skipped cleanly
  end-to-end (`gps_available: false` in `scores.json`, a logged warning in the `full` summary) --
  this is the normal path for this project's actual data, not a degraded fallback.
