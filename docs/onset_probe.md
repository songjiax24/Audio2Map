# Frozen audio onset probe

This diagnostic asks whether physical drum onsets can be read from the frozen v1
audio representation. It does not measure musical interpretation, chart quality,
or the original decoder's ability to use that representation. A successful probe
shows accessible information; an unsuccessful probe also depends on its capacity,
training distribution, and optimization.

## Ownership

- `model/v1/onset.py`: linear or one-hidden-layer `nn.Module` readouts. These are
  separate weights and are never inserted into `V1ChartModel`.
- `eval/v1/onset.py`: hooks into native `generate/v1/stream.encode_context`, physical
  onset bins, subframe reconstruction, and inclusive one-to-one time matching.
  It rejects multiple independent onsets in one frame instead of dropping labels.
- `eval/v1/onset_fit.py`: readout optimization, development-only epoch/threshold
  selection, and test/negative-control scoring.
- `tools/analyze_onset_probe.py`: explicit input manifest, feature cache, and fitting CLI.

Capture uses the actual stem, encoder, phase projection, and combined context. The
phase baseline hooks the phase projection directly; subtracting rounded context
and memory could leak audio-dependent rounding residuals. Capture supports one native
encoder segment per song and rejects multiple segments. Original weights stay in
eval mode, frozen and gradient-free. No reference tokens, chart history, or target
condition vector enter the probe.

## Inputs and clocks

The manifest contains `rows` with `key`, `group`, `split` (`train`, `dev`, `test`),
`kind`, `audio`, `audio_sha256`, `onsets_ms`, and `redlines`. Redlines use the fields
of `osu/v1/timing.Redline`. Audio is mono float WAV at the native sample rate.
Groups cannot cross splits, and every key is unique.

Labels refer to the physical PCM insertion sample, including any input delay.
They are not the desired beatmap timestamps. The head predicts presence plus a
fraction inside the original stem frame. Reported 1/2/5/10 ms tolerances therefore
do not inherit the roughly 11.6 ms frame width. They use floating timestamps and
do not snap predictions to the beat grid.

The 2026-11-08 experiment uses 24 pure-RC training beatmapsets and 8 different
development beatmapsets, plus 8/4 irregular synthetic examples. The previous four
review songs are test-only with both input delays, alongside four new irregular
examples. The original model may already have seen probe-training songs during its
own training; probe development and the four review songs use its validation split.
This is a held-out readout study, not a claim of unseen-song generalization for
the original model. One identical drum waveform is reused throughout, so this
first experiment tests that waveform and timing density, not timbre generalization.

## Run and interpretation

```bash
uv run python tools/analyze_onset_probe.py \
  --manifest .local-data/processed/eval/onset_probe_20261108/inputs_exact.json \
  --checkpoint .local-data/processed/checkpoints/v1_chart/step_6000.pt \
  --output .local-data/processed/eval/onset_probe_20261108/rerun \
  --stage all
```

Use a fresh output directory when inputs, extraction code, or fitting settings change.
The recorded 2026-11-08 results and audit are in that experiment's `overview.md`,
`summary.json`, and `audit.json`; executed source snapshots are preserved under
`final/executed_source/`.

`--stage capture` and `--stage fit` separate expensive extraction from small head
training. The cache/index and fitting protocols freeze hashes and settings.
Raw features may be reused by a different output directory through the same
`feature_index.json`; its paths and hashes identify the original cache.

For every stage, report both linear and 64-hidden-unit heads with seeds 0/1.
Epochs and presence thresholds are selected on development micro F1 at 5 ms;
test data select neither. Controls include independently irregular timing, native
encoding of silent audio, and circularly shifted audio features. For the combined
context control only audio memory is shifted; phase stays at its original time.
The phase-only control remains unchanged when audio features are shifted.
Silence has no true events, so report false positives per minute rather than an
empty-reference F1 as an accuracy score. Offset errors conditioned on true event
presence separate localization from event detection and do not describe the
end-to-end performance of the probe by themselves.

The paired continuation study replays legal source or generated prefixes through
the existing sampler. Conditions, audio, checkpoint, and sampling settings remain
fixed within each comparison. The supplied prefix is excluded from suffix scores.
Its cold-history control restores cursor/absolute positions but feeds just the last
beat marker to a fresh decoder cache; this is an unusual input, not ordinary BOS
generation. A zeroed-audio suffix additionally tests whether a human prefix causes
continued patterns when subsequent acoustic evidence is absent.

Readout success and prefix sensitivity together can narrow the failure to feature
use/generation. Neither establishes which decoder component or training choice
caused it, nor whether the representation is adequate for real music.
