# Architecture notes (maintainability)

Conflict resolution: **code > README > this file**.

## Layering (enforced by import-linter)

Dependency direction is one-way, low → high; each layer may only import layers below it
(`uv run lint-imports`, config in `pyproject.toml`):

```text
cli → eval → train → generate → dataset → model → features → metrics → tokens → grid → osu → utils
```

Vendored MinaCalc (`features/cond/ett/`) and the YAVSRG pattern analyser
(`features/cond/pattern_analyser/`) sit inside the features layer.

Consequences:

- `dataset/` is **torch-free** (pure sampling/building); the torch `Dataset` wrapper is `train/data.py`.
- `model/config.py` owns shared constants (`WINDOW_BARS`, `MAX_DECODER_LEN`, …) so `dataset/` never imports `train/`.
- Inference lives in `generate/` (not `train/`); `generate/service.py` is the single end-to-end
  pipeline used by `audio2map-infer` and the demo backend.
- Chart token grammar lives in `tokens/state.py` (`ChartState`). Generation and validity both
  use it; do not add sampling to `ChartState`.
- `metrics/` is shared by train and eval (token NLL, token accuracy, logit validity). It may import
  `tokens` (and torch) only — not generate, dataset, or model.

## Two `.osu` parsers

| Module | Role |
|--------|------|
| [`audio2map/osu/parser.py`](../audio2map/osu/parser.py) | **Authoritative** for training, tokens, export, eligibility |
| [`audio2map/features/cond/pattern_analyser/osu_parser.py`](../audio2map/features/cond/pattern_analyser/osu_parser.py) | **Vendored** YAVSRG/Converter.fs port for pattern features only |

Do not cross-wire semantics. Changing lane/x parsing in one without the other (and without tests) will desync `cond_vec` analyzer dims from chart notes.

## v0 and v1 timing and audio

v0 `random_full` retains the target audio's `canonical_bpm_norm` and replaces only
the donor's 17 style/difficulty fields. Teacher, generation and demo share
`features/cond/vec.py:with_canonical_bpm`. Generation caches key the actual condition
bytes, so old cases with a different BPM cannot be reused as the corrected input.
Historical baseline reports are not rewritten; verify their original vectors before
attributing a matched/random gap to style conditioning.
New v0 teacher and generation results identify this protocol with
`random_full_policy=donor_style_target_bpm_v1`.

v0 training, export, and eligibility still use `osu/parser.py` (`parse_timing_points` truncates `offset_ms` to `int` and requires eight fields) and `features/audio/loader.py` (libsndfile first, ffmpeg if that fails). v1 does not call those two for timing or PCM.

`Redline` in `osu/v1/timing.py` is the v1 uninherited timing point (a red line). It is not `TimingPoint`: v0 still stores `offset_ms` as `int` and requires eight fields.

v1 authority, until a later task replaces the v0 path:

| Concern | Module |
|---------|--------|
| Red lines (`Redline`, not `TimingPoint`) | `audio2map/osu/v1/timing.py` |
| Chart reject reasons | `audio2map/utils/errors.py` |
| Beat grid and V80 cells | `audio2map/grid/v1/` (`positions.py` holds `V80_DENS`) |
| Labels, vocab, cursor, codec | `audio2map/tokens/v1/` (`encode_chart`, not v0 `encode_notes`) |
| Illegal token | `audio2map/tokens/errors.py` |
| Decode, resample, log-mel | `audio2map/features/audio/v1/` (`frame_index` is a mel frame) |
| Stem convolution | `audio2map/model/audio_stem.py` |

`T_audio` for v1 is `len(y) / sample_rate * 1000` on the PCM from `decode_audio`. There is no second duration via `soundfile.info`. `eval/v1_backend.py` is the v0 eval adapter. It is not this tokenizer, and this round does not rename that file.

The model-independent audio/chart lag diagnostic is owned by
`tools/analyze_audio_offsets.py`. It reuses v1 decoding and the authoritative note
parser, and changes no training or generation defaults. Its sampling, detector
calibration, and uncertainty limits are documented in [audio_alignment.md](audio_alignment.md).

Frozen v1 physical-onset readouts are owned by `eval/v1/onset.py` and
`eval/v1/onset_fit.py`, entered by `tools/analyze_onset_probe.py`. Their small trainable
heads live in `model/v1/onset.py`, separately from chart-model weights. Feature capture
hooks the native inference encoder and phase projection. Split isolation, clocks,
negative controls, and interpretation limits are documented in [onset_probe.md](onset_probe.md).

Whole-chart v1 evaluation lives in `eval/v1/`, entered by `cli/eval_v1.py`
(`audio2map-eval-v1`). It uses BeatGrid endpoint identities and calls
`generate/v1/chart.py` for constrained charts. The unmasked validity diagnostic
shares the production sampling step, but never enters chart metrics. Training
and evaluation both use `train/v1/validation.py` for teacher statistics.
The motif adapter reuses shared fitting/counting algorithms with exact V80
fraction coordinates; it does not call the v0 note-to-tick conversion.
See [v1_eval.md](v1_eval.md) for commands and frozen output contracts.

Optional v1 continuation is owned by `generate/v1/sample.py`, exposed through
`generate_v1_chart(..., prefix=...)`. A legal BOS-led prefix is replayed through the
same decoder and cursor, preserving absolute positions and open holds without RNG draws.
Each call rebuilds the cache under its requested condition. With no prefix the existing
BOS-to-EOS generation path is unchanged. Continuation diagnostics score only the suffix;
supplied human notes must not inflate generation metrics.
`history_start` can restore the cursor from earlier tokens without feeding them into
the decoder. Only the retained suffix warms the fresh KV cache. `model/v1/kv.py` owns
the absolute RoPE offset separately from local ring indexing, including CUDA graph replay;
trimming history must not reuse hidden states carrying information from excluded tokens.

Optional raw-logit observation uses `generate/v1/trace.py` and the same sampler.
`eval/v1/logits.py` records decisions, while `eval/v1/logit_stats.py` interprets
reference onsets and matched history/audio controls. The native legality/temperature
policy remains in `generate/v1/sample.py:sampling_distribution`; analysis reuses it.
See [generation_logits.md](generation_logits.md) for capture and interpretation.

Report version 2 keeps those coordinate and generation owners. `eval/v1/summary.py`
aggregates live and saved artifacts through one path; `rescore.py` verifies and reuses old
tokens without resampling. `statistics.py` owns chart-level uncertainty and paired responses,
`archive.py` checks that saved records cover every frozen case with its original condition.
`comparison.py` compares frozen runs with matching audio provenance for each mode.
`probes.py` owns separate audio-shift and isolated-condition experiments. ROW detail counts reuse `metrics/token.py`; motif fitting,
frequency distance, and error contributions remain in the shared `eval/motif*.py` modules.
`eval/v1/reporting/` owns Markdown, HTML, and SVG rendering. It consumes summaries and
records prepared by the evaluation workflows. Metric computation and summary aggregation
have no dependency on the renderers.
`eval/v1/protocol.py` validates optional frozen probe requests and calibrates pattern levels
from verified train-only observations using the existing condition metadata implementation.
Compatible completed probe cases can be copied into a new journal with source provenance;
the original run is never rewritten.
The CLI shares probe config defaults and the motif vocabulary size constant. Shared token prefixes
remain in `tokens/vocab.py`; the report uses `utils/time.py:MS_PER_SECOND` for unit conversion.
The new active-gate decoder parity test mirrors its package under `tests/model/v1/`.

## One sample builder

[`audio2map/dataset/sample.py`](../audio2map/dataset/sample.py) exposes the single
`build_sample(source: Path | ChartBundle, ...)` used by both the train path (preloaded
`ChartBundle` from `dataset/bundle.py`) and eval/one-off paths (direct `Path`).
`tests/dataset/test_sample_parity.py` locks the two call shapes to identical outputs;
if train and eval diverge, add a failing test before "fixing" only one side.

## Data roots

- Set `AUDIO2MAP_DATA_ROOT` for raw grids / formal checkpoints. If unset, the library
  warns and uses repo-local `.local-data/` (gitignored). Data-backed tests skip when
  `raw/` is missing.
- Repo-local `checkpoint/` and `chart_meta/` are optional demo copies (gitignored).
- See [`audio2map/utils/paths.py`](../audio2map/utils/paths.py).

## Configs are real

`configs/*.yaml` are loaded by every CLI entry point via `--config` (YAML supplies
argparse defaults; CLI flags win). Unknown keys are ignored with a stderr warning so
configs may carry informational values. See [`audio2map/cli/common.py`](../audio2map/cli/common.py).

## Windows

Training and generate windows are 16 bars (`WINDOW_BARS` in `model/config.py`).
Inference overlap is the same 8 context + 4 keep + 4 future (`OverlapConfig`;
`configs/data/v2.yaml` records the values but they are not CLI-tunable).
Masked CE lives in `train/step.py`, not on the `nn.Module`.
