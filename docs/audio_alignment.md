# Audio / chart alignment diagnostic

`tools/analyze_audio_offsets.py` measures whether changes in the original audio
systematically follow chart note heads. It does not use model outputs, editor
waveforms, hitsounds, or game playback offsets, and does not retime any input.
Positive lag means that the audio feature occurs later than the note head.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python tools/analyze_audio_offsets.py \
  --groups 400 --workers 3 \
  --output .local-data/processed/analysis/audio_offsets_run
```

The default population is the train and validation charts in the frozen split
manifest. Test charts are not inspected. Sampling is uniform over beatmapset
groups, with one randomly selected audio file and up to three randomly selected
charts for that audio. This is not a uniform sample of all chart difficulties.
Deduplicate exact audio hashes and account for groups when estimating uncertainty.

The script decodes audio with the production v1 decoder and rejects damaged
timelines. It samples positive spectral changes at about 1 ms resolution using
two centred Hann window lengths. Known synthetic digital attacks calibrate each
detector's time convention. Each distinct head time counts once, including LN
heads; LN tails and simultaneous extra keys do not add observations. Only heads
at least 80 ms from neighbouring heads are used. This deliberately sacrifices
coverage to reduce confusion with neighbouring events.

For each chart, head-relative profiles are scanned over -80 to +80 ms. Profiles
are normalized per head and averaged within six time blocks. A block bootstrap
measures variation across the chart. Too few anchors, weak or competing peaks,
unstable block estimates, boundary peaks, or disagreement between window lengths
leave the chart unresolved. The JSON protocol records every threshold before
measurement. Unresolved and failed cases must stay visible in coverage reports.

`protocol.json` freezes selection, settings, calibration, and diagnostic source
hash. `groups.jsonl` contains incremental per-group results, file hashes, both
alignment curves, local estimates, confidence intervals, and failure reasons.
Rerunning the same command checks saved audio and chart hashes before reusing a
completed group. Failed groups are retried. Changed settings, source code, or
completed input files require a new output directory.

The predefined 10–30 ms and 15–25 ms bands describe proximity to 20 ms, not
proof that applying exactly 20 ms is correct. Spectral changes are attack proxies:
instrument envelopes, mixing, and what the mapper chose to represent can shift
them. Calibration on synthetic attacks does not remove these uncertainties.
Bootstrap intervals describe within-song sampling variation, not all detector
or annotation error. Raw waveform audits and alternate detectors are necessary
before turning the observed distribution into a preprocessing rule.

A consistent positive lag motivates advancing audio for a new training
experiment. It does not justify the same intervention on an existing checkpoint,
which may already have adapted to the original lag. Any eventual compensation
must share one train/inference implementation and define its interaction with
rate augmentation. No compensation is enabled by this diagnostic.
