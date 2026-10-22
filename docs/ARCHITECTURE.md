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
