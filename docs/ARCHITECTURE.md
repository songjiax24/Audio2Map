# Architecture notes (maintainability)

Conflict resolution: **code > README > this file**.

## Layering (enforced by import-linter)

Dependency direction is one-way, low → high; each layer may only import layers below it
(`uv run lint-imports`, config in `pyproject.toml`):

```text
cli → eval → train → generate → dataset → model → features → tokens → grid → osu → utils
```

Vendored MinaCalc (`features/cond/ett/`) and the YAVSRG pattern analyser
(`features/cond/pattern_analyser/`) sit inside the features layer.

Consequences:

- `dataset/` is **torch-free** (pure sampling/building); the torch `Dataset` wrapper is `train/data.py`.
- `model/config.py` owns shared constants (`WINDOW_BARS`, `MAX_DECODER_LEN`, …) so `dataset/` never imports `train/`.
- Inference lives in `generate/` (not `train/`); `generate/service.py` is the single end-to-end
  pipeline used by `audio2map-infer` and the demo backend.

## Two `.osu` parsers

| Module | Role |
|--------|------|
| [`audio2map/osu/parser.py`](../audio2map/osu/parser.py) | **Authoritative** for training, tokens, export, eligibility |
| [`audio2map/features/cond/pattern_analyser/osu_parser.py`](../audio2map/features/cond/pattern_analyser/osu_parser.py) | **Vendored** YAVSRG/Converter.fs port for pattern features only |

Do not cross-wire semantics. Changing lane/x parsing in one without the other (and without tests) will desync `cond_vec` analyzer dims from chart notes.

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
