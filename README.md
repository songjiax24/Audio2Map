# Audio2Map

Conditional generative model for **osu!mania 4K** charts:

```text
P(chart | audio, BPM, offset, condition_vector)
```

Phase 1 **training** filter: **constant BPM**, **meter = 4**.  
Collector stores **all mania 4K** from Sayobot ranked/approved sets (variable BPM included); eligibility is applied later (`audio2map-meta --eligible-only`, training).  
This is a **generative** task (many valid charts per audio); training metrics like token accuracy are diagnostics only, not the final quality target.

> **Source of truth:** the Python package under `audio2map/` and the `audio2map-*` CLI entry points (`pyproject.toml [project.scripts]`).  
> Conflict: **code > this README > `docs/`**.

---

## Documentation map

| Document | Role |
|----------|------|
| **[README.md](README.md)** (this file) | Install, CLI, current model numbers |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layering, dual parsers, sample builder, paths |
| [docs/chart_tokens.md](docs/chart_tokens.md) | Token cheat sheet (vocab, loss mask) |
| [docs/data_collection.md](docs/data_collection.md) | Sayobot collector (only if collecting) |
| [configs/README.md](configs/README.md) | YAML `--config` files |
| [demo/README.md](demo/README.md) | Web demo |

---

## Repository layout

```text
audio2map/                    # Python package (dependency direction is one-way, low → high)
├── utils/                    # paths.py (AUDIO2MAP_DATA_ROOT)
├── osu/                      # authoritative .osu parser, schema, export (ms, not quantized)
├── grid/                     # CanonicalTiming, tick/bar geometry, TickNote
├── tokens/                   # ROW vocab, encode, decode
├── features/                 # tick audio grid + cond_vec (meta / SR · MinaCalc · patterns)
├── dataset/                  # torch-free: filter, windows, bundle, sample (build_sample)
├── model/                    # AudioChartModel (enc_dec), rotary, config (MAX_DECODER_LEN …)
├── train/                    # data.py, step.py (masked CE), loop.py
├── generate/                 # decode, overlap inference, service (single end-to-end pipeline)
├── eval/                     # teacher / generate / note-match metrics
└── cli/                      # argparse: meta/grid/train/eval/infer (--config YAML)

scripts/                      # pipeline (not core library): collect, overfit, batch_pack, formal pipeline
tools/                        # diagnostics: analyze_*
demo/                         # Web demo source (backend + frontend)
tests/                        # pytest, mirrors package layout (tests/osu, features, dataset, …)
docs/                         # architecture, tokens, collector
configs/                      # YAML configs — actually loaded via --config
```

**Code vs data:** git holds **code only** (`audio2map/`, `scripts/`, `tools/`, `demo/` source, `tests/`, `docs/`, `configs/`, `uv.lock`).

| On disk (gitignored) | Role |
|----------------------|------|
| `checkpoint/` | Optional local demo fallback (`step_200000.pt`); demo prefers `$AUDIO2MAP_DATA_ROOT/processed/checkpoints/formal_enc_dec_v3/` |
| `chart_meta/` | Optional local `manifest.jsonl` for demo — or `AUDIO2MAP_CHART_META_MANIFEST` / `$AUDIO2MAP_DATA_ROOT/chart_meta/` |
| `$AUDIO2MAP_DATA_ROOT/` | Raw sets, audio grids, formal checkpoints (see below) |

Do not commit weights, manifests, `node_modules/`, or `demo/.tools/`.

---

## Data disk layout

Set `AUDIO2MAP_DATA_ROOT` to the data disk. If unset, the library warns and uses
repo-local `.local-data/` (gitignored).

```text
$AUDIO2MAP_DATA_ROOT/
├── raw/                      # {set_id}/*.osu + referenced audio
├── processed/                # audio_grid, checkpoints, generated, logs
│   ├── audio_grid/           # {stem}.npy + .json  (tick log-mel, spec v3)
│   ├── checkpoints/          # formal_enc_dec_v3/ (current), older trial dirs
│   ├── logs/
│   └── generated/
├── chart_meta/               # manifest.jsonl (SR, MSD, pattern stats)
└── .collector/
```

---

## Setup (uv)

```bash
apt-get update && apt-get install -y ffmpeg libsndfile1

curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/Audio2Map
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev,train,demo]"
# PyTorch CUDA wheel (optional): https://pytorch.org/get-started/locally/
# Example: uv pip install torch --index-url https://download.pytorch.org/whl/cu128

export AUDIO2MAP_DATA_ROOT=/path/to/audio2map_data   # required for real data
pytest tests/osu tests/features -q
```

Copy [.env.example](.env.example) to `.env` for local paths and collector contact
(`AUDIO2MAP_CONTACT`). Do not commit `.env`.

**Web demo** (local checkpoint + chart_meta): see [demo/README.md](demo/README.md).

```bash
# backend
uv run --extra train --extra demo uvicorn demo.backend.app:app --host 0.0.0.0 --port 8000
# frontend
cd demo/frontend && npm install && npm run dev
```

---

## CLI reference

Installed via `pyproject.toml [project.scripts]` (or `python -m audio2map.cli.<name>`).
All entries accept `--config <yaml>` for defaults; CLI flags override the file.

| Command | Purpose |
|---------|---------|
| `audio2map-grid precompute` | Build `processed/audio_grid/` caches (spec v3, 128-d) |
| `audio2map-grid audit` | Validate caches; `--fix` removes corrupt files |
| `audio2map-meta` | Build `chart_meta/manifest.jsonl` |
| `audio2map-train` | Formal multi-chart training (`AudioChartModel`) |
| `audio2map-eval` | Teacher / infer metrics |
| `audio2map-infer` | Full-chart AR generate → export `.osu` |

Pipeline scripts (not installed entry points): `python -m scripts.collect.cli`,
`scripts/run_formal_pipeline.sh`, `scripts/train_debug_overfit.py`,
`scripts/batch_pack_osz.py`.

Diagnostic tools live in `tools/` (run as plain scripts): `analyze_dataset.py`,
`analyze_timing.py`, `analyze_row_token_stats.py`, `analyze_tick_quantization.py`,
`analyze_window_token_len.py`.

---

## Typical workflow

### 1. Collect (optional if raw/ already populated)

```bash
python -m scripts.collect.cli --target 1000
python -m scripts.collect.cli --status
```

### 2. Precompute audio grids

```bash
audio2map-grid audit
audio2map-grid precompute   # skips valid; for new raw sets
```

### 3. Train (formal)

```bash
audio2map-train --config configs/train/train_multi.yaml
# defaults: 200k steps, batch 16, d_model=512, enc=4/dec=6/heads=8,
#           samples_per_chart=4, bf16 → processed/checkpoints/formal_enc_dec_v3/
```

Or explicit:

```bash
audio2map-train \
  --max-steps 200000 \
  --batch-size 16 \
  --samples-per-chart 4 \
  --d-model 512 \
  --out "$AUDIO2MAP_DATA_ROOT/processed/checkpoints/formal_enc_dec_v3"
```

- Grid required by default (`--allow-missing-grid` to opt out)
- Architecture is **enc_dec only** (no `--architecture` / bar-pooling flags)
- No `--resume`: each run starts at step 0

### 4. Evaluate / generate

```bash
audio2map-eval --mode teacher \
  --checkpoint .../step_200000.pt \
  --osu "path/to/chart.osu" --start-bar 8

audio2map-infer \
  --checkpoint .../step_200000.pt \
  --osu "path/to/template.osu" \
  --out "path/to/output.osu"
# overlap: 16 bars = 8 context + 4 keep + 4 future (fixed, same as training)
```

Pack `.osz`: zip the generated `.osu` + its `AudioFilename` audio from the same set folder, or use the web demo.

---

## Model summary (current code)

| Item | Value |
|------|-------|
| Class | `AudioChartModel` (`architecture="enc_dec"`) |
| Formal size | `d_model=512`, enc=4, dec=6, heads=8 (`audio2map-train` defaults) |
| Vocab | **821** (4 special + 192 POS + 625 ROW) |
| Training window | **16 bars** = **3072** ticks (`WINDOW_BARS`) |
| Inference overlap | **8 context + 4 keep + 4 future** = 16 |
| Max decoder len | **2048** |
| Audio | **128-d** log-mel / tick @ 22.05 kHz; STFT `n_fft=1024`, `hop=128`; **spec v3**; linear interp onto tick starts |
| Condition | **18-d** `cond_vec` (dims 0–16 style/difficulty, dim 17 `canonical_bpm_norm`) — not tokenized |
| Versions | `TOKENIZER_VERSION=1`, `COND_VEC_VERSION=1`, audio **spec v3** (`AUDIO_FEATURE_SPEC_VERSION`) |
| Encoder pos | RoPE + learned `pos_in_bar` |
| Cond injection | MLP → broadcast-add on all encoder/decoder tokens |
| Loss | Masked cross-entropy (teacher forcing) |

`cond_vec` names (fixed order): `COND_VEC_NAMES` in `audio2map/features/cond/vec.py`. Matching / demo sliders use ChartMeta originals (`USER_COND_SOURCE_FIELDS`: `official_sr`, analyzer ratios, `msd_*`). `build_cond_vec` maps those to the vector (`official_sr/10` and `msd_*/40`, clip 1.5). `offset_ms` is **not** in `cond_vec`. Do not reorder `COND_VEC_NAMES` without bumping `COND_VEC_VERSION`.

---

## Tests

```bash
pytest tests/ -q
```

Without `AUDIO2MAP_DATA_ROOT` / raw charts, data-backed tests **skip** automatically.  
Optional marker for CI-only filtering: `-m "not requires_data"` (same outcome when data is absent).

Tests mirror the package layout: `tests/osu/`, `tests/tokens/`, `tests/features/`,
`tests/dataset/`, `tests/model/`, `tests/train/`, `tests/generate/`, `tests/eval/`,
`tests/cli/`, `tests/collect/`, `tests/demo/`, plus `test_paths.py`.
Import layering is enforced by import-linter (`uv run lint-imports`, see `pyproject.toml`).

---

## What was removed

- **v1** 10 ms mel + sparse-event pipeline (no longer in this repo)
- **`PrefixLMAudioChartModel`**, bar-level audio pooling, variable `window_bars_choices` as the formal path
- Older checkpoint dirs such as `train_v1/` / `trial_multi_v2/` are obsolete if still present on the data disk

---

## License

Code: project authors. Beatmap assets: respective creators / osu! rights holders.
