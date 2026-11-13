# Whole-chart v1 evaluation

`audio2map-eval-v1` implements `v1_design.md` §9.2. The existing
`audio2map-eval` command and `eval/v1_backend.py` keep their v0 meaning.
Commands below also work as `uv run --extra train python -m audio2map.cli.eval_v1` before
reinstalling the project's console entry points.

Evaluation is a set of capability tests, not an automatic chart-quality score.
Higher reference F1 is positive evidence of reference reproduction. Multiple
reasonable charts can share the same music and conditions, so F1 is interpreted
alongside condition response, distributions, diversity, and structural diagnostics.

## Prepare the motif vocabulary

Use the current v1 training split. The vocabulary has its own version and
tokenizer identity; a v0 motif list is rejected.

```bash
audio2map-eval-v1 --mode fit-motifs --split train \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/motif_train"
```

This freezes `pool.json`, `donors.json`, and `motifs.json`. The donor map is
unused during fitting. `--limit` or `--osu` makes a diagnostic vocabulary,
recorded by `corpus.subset`; use all train charts for a formal vocabulary.

## Evaluate a checkpoint

```bash
audio2map-eval-v1 --config configs/eval/v1.yaml \
  --checkpoint "$AUDIO2MAP_DATA_ROOT/processed/checkpoints/v1_chart/step_6000.pt" \
  --motif-vocab "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/motif_train/motifs.json" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val"
```

Select the partition explicitly with `--split train|val|test`, including
when supplying `--osu`. Relative `--osu` paths are relative to `raw/` and
must belong to that partition. The default manifest is
`splits/beatmapset_v1_drop2509435.json`. Override it with `--split-manifest`.
`--raw-root` can override the raw directory without changing other data paths.

`--mode teacher` does not need a motif vocabulary. `--mode constrained` runs
chart metrics. `--mode unconstrained` runs only the legality diagnostic.
`--mode all` runs all three. `--limit N` selects a deterministic sample using
`--seed`, before the v1 input gate. Rejected charts and their reasons are
reported. A limited or explicit pool is marked as a subset.

Production CUDA generation uses bf16, matching inference. `--precision fp32`
is supported for teacher-only CUDA evaluation; generation rejects it rather
than silently ignoring it. CPU execution uses fp32. `run.json` records both
the requested precision and the effective teacher/generation precisions.

`--seeds 0` is useful for a timing probe. The default seeds 0–4 produce
diversity pairs on `matched_full`. Other constrained scenarios and all
unconstrained scenarios use the first sampling seed. Partial masks use
`--seed`, target path, and scenario, and stay fixed across sampling seeds.

## Outputs and definitions

- `run.json`: checkpoint, split, selected chart and audio SHA256, versions,
  execution device, sampling settings, and subset flags.
- `pool.json`, `donors.json`, `conditions.json`: frozen accepted inputs,
  rejections, target-to-donor mapping, actual values, and actual masks. The
  donor universe is exactly the accepted pool; self-draws are counted.
- `gate/`: this run's derived input-gate cache. Separate runs do not write
  each other's gate files or share decisions between raw directories.
- `metadata.jsonl`: rate-1 input conditions prepared by the existing parallel
  ChartMeta builder, scoped to this run and raw directory.
- `records.jsonl`: flushed chart/scenario/mode records. A failed run keeps its
  completed records; it does not receive a complete summary.
- `summary.json`: teacher, constrained, unconstrained, diversity, distribution,
  motif, wall time, and peak allocated CUDA memory. Scenarios stay separate.
  `summary.md` renders the main tables for reading. The renderers live in
  `audio2map/eval/v1/reporting/` and consume the computed summary and records.
  Reference plausibility uses the same decoded and floored representation as
  generated notes. Each reference includes the audio duration used to build its grid.
- `generated/`: constrained `.osu` files and their original token sequences.
  Audio remains in the input corpus and is symlinked beside each exported chart.

An output directory belongs to one configuration, checkpoint, and frozen input
pool. Repeating identical inputs recomputes results. Changing frozen inputs
requires another output directory. There is no implicit reuse of a previous
checkpoint's generations.

The evaluation verifies the motif vocabulary's raw root and split manifest,
and checks that every fitting chart belongs to the train partition. Generation
and teacher forcing decode the hashed audio sources directly. Rescoring uses
the saved reference grid duration for generated tokens. Older reports fall
back to the source gate duration or a decode after verifying available source
hashes. Runs created before direct decoding was recorded remain readable by
rescoring if the required generation artifacts are present.
Continuing generation from those runs requires a new output directory.

Use `--resume` with the same command and output directory to reuse completed
records after interruption. Frozen settings still have to match. Generated
token and `.osu` checksums are verified before reuse; missing or altered
artifacts are recomputed. An incomplete final journal line is discarded, while
corrupt complete records raise an error. `reused_records` reports the reuse
count. Without `--resume`, an identical run is recomputed. Archive consumers
check the exact set of frozen chart, scenario, mode, and seed cases and verify
that each record agrees with its frozen condition.

Endpoint identity is `(lane, section, beat_k, numerator, denominator, kind)`.
Kinds are tap, hold head, and hold tail. Millisecond rounding never changes
these identities. Distributions include endpoint denominator counts, per-lane
kind counts, hold spans in legal grid-point ordinals, and exact beat spans.
Plausibility compares adjacent objects within each lane. Concurrent
`next.start <= prev.end + 2 ms` and almost-concurrent
`abs(next.start - prev.end) < 10 ms` are counted independently. Floored
duplicates and nonpositive holds are retained.

Motif extraction reads the original ROW tokens, including ongoing hold states.
It reuses the existing lane/hand/row pooling, mirrored right hand, Top-K fitter,
and frequency distances. Positions encode the BeatGrid beat index plus its
V80 fraction in `lcm(V80_DENS)` integer units per beat. This is an exact
serialization of the fractions, not v0 tick quantization. A residual beat still
advances one BeatGrid beat index. Vocabulary fitting uses train only. All
vocabulary entries, including zero counts, enter frequency distances.

Teacher uses the same whole-chart validator as training. It reports per-chart
and token-weighted NLL/accuracy and raw-logit legality. Per-type perplexity is
derived from per-type NLL. Overflowing perplexity is `null`, not JSON Infinity.
Condition errors use the 18 original ChartMeta fields and the condition actually
sent to the model; missing or masked values are `null`, never zero error.

Unconstrained sampling shares the production decoder step but draws from all
693 tokens, including PAD and BOS. It stops at the first illegal token and
records its position. A run exceeding the grid-derived legal sequence bound
without EOS is truncated. These results never enter chart metrics.

## Expanded capability reports (version 2)

Matching now includes lane-aware and lane-agnostic event and complete-note
identities. A complete hold matches only when both its endpoints match as a pair.
Dropping the lane preserves multiplicities of simultaneous events. Legacy endpoint
`all`, `by_kind`, and `by_lane` outputs retain their definitions.

Teacher additionally reports ROW lane accuracy and tap/head/tail recall, reusing
the existing generic ROW counter. Historical teacher records without these counts
remain unavailable until refreshed; they are not treated as zero.

Matching includes micro F1, sample distributions, chart distributions, and chart
mean confidence intervals. Seeds are averaged within a chart before p10/p50/p90
or bootstrap comparisons. Bootstrap uses 2,000 draws, seed 20261106, and a 95%
interval. Intervals describe these charts and fixed generations, not all possible
sampling variability. p10–p90 is a distribution range, not a confidence interval.

Distribution counts add exact positions, complete-note counts, and ROW states.
Motif adds per-length additive contributions to MAE, largest gaps, vocabulary
coverage by length, and redline-crossing window counts. `length_buckets` keeps its
old within-bucket mean definition; `by_length.mae_contribution` divides by the whole
scope vocabulary size and sums to total MAE. Coordinates and frozen vocabularies
have not changed. Crossing counts audit the beat-index treatment of residuals;
they do not claim to correct its musical-time interpretation.

Joint-condition response compares matched_full/random_full at the first seed,
excluding donor self-draws and unavailable/unchanged target fields. It reports
direction agreement, slope, observed changes, and per-chart pairs. The entire
condition vector changes together; this does not isolate a single field's effect.

## Reuse existing generations

```bash
audio2map-eval-v1 --mode rescore \
  --source-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val32" \
  --motif-vocab "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/motif_train/motifs.json" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val31_refreshed" \
  --refresh-teacher --device cuda --html
```

Rescoring verifies chart/audio inputs and generated token/`.osu` hashes, then
uses the same metrics and aggregation as a live run. It never samples or alters
the source directory. The new directory records source hashes. Legacy run files
without a vocabulary path require `--motif-vocab` with the original hash.
`--refresh-teacher` performs new forward passes for existing teacher cases
and records the current audio source, FFmpeg, soxr, and mel normalization
separately. Saved generated tokens are still scored on their reference grid.
Without a refresh, saved teacher results and their provenance are retained.
The elapsed time measures the rescoring pass.

`--html` writes standalone SVG figures embedded in `index.html`; it requires the
optional plotting dependency: `uv sync --extra train --extra eval`. Without it,
JSON and Markdown evaluation still work. Plotting availability is checked before
long computations start. HTML includes observed counts, matching distributions,
condition response, human/generated distributions, motif errors and coverage,
diversity, legality, and links to empty/conflicting/measurement-failure samples.

## Paired checkpoint comparison

```bash
audio2map-eval-v1 --mode compare \
  --source-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step4000_val31_refreshed" \
  --compare-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val31_refreshed" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/comparison_refreshed"
```

Both runs must have version-2 scores and exactly matched frozen charts, audio,
conditions/masks, seeds, temperature, context settings, and motif hash. For each
compared mode, effective precision and audio provenance must agree. Older
teacher refreshes without their own decoder provenance need a new refresh
before comparison.
Differences are **right minus left**, paired by chart after within-chart seed
averaging. Missing measurements are excluded with pair counts preserved. There
is no composite rank: negative NLL/error differences and positive F1/accuracy
differences have different interpretations.

## Controlled probes

```bash
audio2map-eval-v1 --mode probes \
  --source-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val32" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_probes_v2" \
  --probe-teacher-charts 16 --probe-charts 4 --probe-seeds 0 1 --device cuda --html
```

These experiments have a separate frozen plan and journal. They do not change
the five standard scenarios, production sampler, or temperature. Use `--resume`
with identical settings after an interruption.

Audio alignment: circularly shift PCM by 0 or ±5 seconds, capped at one quarter of
clip length. Preserve duration, reference tokens, redlines, and conditions. Record
the actual shift in samples and clear the original mel cache key before rebuilding
features. Teacher uses matched_full and none. Free generation uses matched_full
and the first seed. Positive shifted-minus-correct NLL or negative shifted-minus-
correct reference F1 supports an audio-alignment contribution. It is sensitivity
evidence, not a direct music-quality judgment; the wrap boundary is part of the
perturbation.

Generated shift cases link to a float WAV containing the shifted input. Resume
and cross-run reuse verify it alongside the chart and tokens. Earlier probe
runs without decoder-version provenance require a fresh probe directory.

Isolated requests: retain only official SR at 2/4/6, or LN proportion at 0/0.5/1;
mask the other 17 fields to avoid conflicting correlated requests. Average seeds
within each chart and report measured curves, coverage, centered regression
slopes, and monotonicity. A one-field mask is sparser than typical training
partial masks. This tests sparse-condition ability; failure does not establish
that full conditioning fails. Level choices are controlled requests, not a sample
from the data's condition distribution.

Default selection is deterministic within the source run's accepted pool. Small
probe sets are explicitly reported; neither shift tests nor condition sweeps are
interpreted as an overall quality score.

### Pattern calibration and finer audio shifts

```bash
python tools/compute_v1_probe_protocol.py \
  --source-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val32" \
  --training-vocab "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/motif_train/motifs.json" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/pattern_calibration_v3" \
  --charts 512 --seed 0 --workers 4

audio2map-eval-v1 --mode probes \
  --source-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_val32" \
  --out "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_pattern_fine_v3" \
  --probe-protocol "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/pattern_calibration_v3/protocol.json" \
  --probe-reuse-run "$AUDIO2MAP_DATA_ROOT/processed/eval/v1/step6000_probes_v2" \
  --probe-teacher-charts 16 --probe-charts 4 --probe-seeds 0 1 --device cuda --html
```

Calibration checks the frozen vocabulary corpus against the actual train split and
chart hashes, samples at most 512 accepted charts, and uses the production v1 metadata
calculator. Each pattern gets zero plus the 25th/75th percentiles of its positive
observations. Duplicate levels are removed; all-zero or missing fields are skipped.
The protocol preserves the sample list, source hashes, zero/positive counts and measured
distributions. These six shares measure per-class interval-union coverage and can overlap;
they are never normalized to sum to one. The sweep isolates one field at a time with the
same sparse-mask limitation described above.

The generated protocol requests ±50/100/250/500 ms in addition to zero and ±5 s.
Report plots zoom into the smaller offsets. Actual shifts are rounded to samples and
still capped by duration: at 22050 Hz, a requested 50 ms becomes 1102 samples, about
49.977 ms. Both requested and actual shifts are saved.
Audio generation also reports per-case note/token counts, empty outputs and metadata
errors, with links to generated cases. Empty charts remain in F1; unavailable component
measurements remain missing. Token count alone is not temporal coverage or truncation.

`--probe-reuse-run` can be repeated for complete prior probe runs. Reuse checks the
source/checkpoint hashes, inference settings, exact per-case inputs and generation
artifact checksums. Copied records carry their source; the target journal remains
complete and the old run is untouched. The new run's elapsed time excludes the original
cost of reused cases. `--resume` continues the target with an identical frozen plan.
