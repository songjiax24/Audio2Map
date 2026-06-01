# Audio2Map

Conditional generative model for **osu!mania 4K** charts:

```text
P(chart | audio, BPM, offset, condition_vector)
```

Phase 1 scope: **constant BPM**, **meter = 4**, Sayobot ranked/approved mania sets.  
This is a **generative** task (many valid charts per audio); training metrics like token accuracy are diagnostics only, not the final quality target.

---

## Documentation map

> **Only [docs/OVERVIEW.md](docs/OVERVIEW.md) is confirmed by sjx.** Other docs and the codebase may not match it.

| Document | Status |
|----------|--------|
| **[docs/OVERVIEW.md](docs/OVERVIEW.md)** | **sjx confirmed — authoritative** |
| [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) | unverified agent notes |
| [docs/V2_MASTER_SPEC.md](docs/V2_MASTER_SPEC.md) | unverified |
| [docs/v2_spec.md](docs/v2_spec.md) | unverified |
| [docs/PROJECT_CONVERSATION_LOG.md](docs/PROJECT_CONVERSATION_LOG.md) | conversation history |
| [docs/data_collection.md](docs/data_collection.md) | unverified |
| [configs/README.md](configs/README.md) | reference YAML (not loaded by scripts) |

---

## Repository layout

```text
audio2map/                    # Python package
├── audio/                    # tick_features.py (v2, 142-d/tick @ 22.05 kHz)
├── data/                     # datasets, audio_grid cache, chart bundles
├── difficulty/               # chart_meta, SR, MSD, pattern analyser hooks
├── eval/                     # teacher/generate metrics
├── legacy/                   # archived v1 sparse events (10 ms) — not used in training
├── osu/                      # parser, row_tokens (v2), export
├── pattern_analyser/         # vendored YAVSRG prelude features
├── training/                 # AudioChartModel, decode, inference
└── utils/paths.py            # AUDIO2MAP_DATA_ROOT layout

scripts/                      # CLI entrypoints (see below)
tests/                        # pytest
docs/                         # specs + handoff
configs/                      # reference hyperparameters
```

**Code vs data:** git holds code only. Datasets, grids, checkpoints live on the **data disk** (see below).

---

## Data disk layout

Default: `export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data`

```text
$AUDIO2MAP_DATA_ROOT/
├── raw/                      # {set_id}/audio.mp3 + *.osu  (~22G)
├── processed_v2/             # ACTIVE v2 artifacts
│   ├── audio_grid/           # {stem}.npy + .json  (precomputed tick audio)
│   ├── checkpoints/          # train_v1/ (formal), trial_multi_v2/ (legacy)
│   ├── logs/                 # train / precompute logs
│   └── generated/            # infer exports (.osu, .osz)
├── chart_meta/               # manifest.jsonl (SR, MSD, pattern stats)
├── .collector/               # download state + logs
└── processed/                # DEPRECATED v1 mel cache (~10G) — safe to delete on disk
```

Nothing under `raw/`, `processed_v2/`, or `chart_meta/` is committed to git.

---

## Setup (5090 / fresh instance)

```bash
apt-get update && apt-get install -y ffmpeg libsndfile1

curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/audio2map
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev,train]"
# PyTorch: pick CUDA wheel for your GPU — https://pytorch.org/get-started/locally/
# Example (5090): uv pip install torch --index-url https://download.pytorch.org/whl/cu128

export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
pytest tests/test_row_tokens_v2.py tests/test_v2_pipeline.py -q
```

Copy [.env.example](.env.example) for collector etiquette overrides.

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `collect.py` | Download mania 4K sets via Sayobot (`audio2map-collect`) |
| `precompute_audio_grid.py` | CPU: build `processed_v2/audio_grid/` caches |
| `audit_audio_grid.py` | Validate caches; `--fix` removes corrupt files |
| `compute_chart_meta.py` | Build `chart_meta/manifest.jsonl` |
| `analyze_dataset.py` | Eligibility + token-length stats over `raw/` |
| `analyze_timing.py` | BPM/timing distribution scan |
| `train_v2.py` | Multi-chart training (`AudioChartModel`) |
| `train_debug_overfit.py` | Single-chart/window memorization debug |
| `eval_v2.py` | Roundtrip / teacher / generate / infer metrics |
| `infer_v2.py` | Full-chart AR generate → export `.osu` |
| `analyze_row_token_stats.py` | ROW/hold token frequency stats (diagnostic) |

---

## Typical workflow

### 1. Collect (optional if raw/ already populated)

```bash
python scripts/collect.py --target 1000
python scripts/collect.py --status
```

### 2. Precompute audio grids

**Status (2026-06-01): complete** — 3,567 valid caches; all 11,927 eligible charts trainable.

Maintenance only:

```bash
python scripts/audit_audio_grid.py             # expect invalid=0
python scripts/precompute_audio_grid.py        # skips valid; for new raw sets only
```

### 3. Train (formal)

```bash
python scripts/train_v2.py
# defaults: 11927 charts, audio_pooling=tick, 50000 steps → checkpoints/train_v1/
```

Or explicit:

```bash
python scripts/train_v2.py \
  --steps 50000 \
  --batch-size 8 \
  --samples-per-chart 2 \
  --audio-pooling tick \
  --out "$AUDIO2MAP_DATA_ROOT/processed_v2/checkpoints/train_v1"
```

- **`require-grid`:** default **on** (use `--allow-missing-grid` to opt out)
- **`audio_pooling=tick`:** formal default; legacy `bar` only for old checkpoint reproduction
- **No `--resume`:** each run starts at step 0
- **Preload:** ~1–2 h for 11k charts, then ~25 step/s on 5090

Legacy trial checkpoints: `checkpoints/trial_multi_v2/` (bar-pool, ≤6159 charts — **not** current target).

### 4. Evaluate / generate

```bash
# Teacher forcing on a window (diagnostic)
python scripts/eval_v2.py --mode teacher \
  --checkpoint .../step_20000.pt \
  --osu "path/to/chart.osu" --start-bar 8

# Full chart → .osu
python scripts/infer_v2.py \
  --checkpoint .../step_20000.pt \
  --osu "path/to/template.osu" \
  --out "path/to/output.osu"
```

Pack `.osz`: zip the generated `.osu` + its `AudioFilename` audio from the same set folder.

---

## Model summary

| Item | Value |
|------|-------|
| Class | `AudioChartModel` |
| Parameters | ~3.7M |
| Vocab | 821 tokens (4 special + 192 POS + 625 ROW) |
| Window | 8 bars, random start each sample |
| Audio | 142-d per tick → **tick-level** encoder prefix (`audio_pooling=tick`; legacy `bar` for old ckpt) |
| Condition | 23-d `cond_vec` (SR, holds, pattern stats, BPM) — not tokenized |
| Loss | Masked cross-entropy (teacher forcing) |

See [docs/v2_spec.md](docs/v2_spec.md) for token grammar and [docs/AGENT_HANDOFF.md](docs/AGENT_HANDOFF.md) for runtime state.

---

## Tests

```bash
pytest tests/ -q
```

Key modules: `test_row_tokens_v2.py`, `test_v2_pipeline.py`, `test_eval.py`, `test_decode_inference.py`.

---

## What was removed (v1)

The old **10 ms mel + sparse event** pipeline (`scripts/preprocess.py` → `processed/`) is **removed from this repo**.  
If `processed/` still exists on the data disk, it is unused and may be deleted to free ~10 GB.  
Sparse-event code lives under `audio2map/legacy/` for tests only.

---

## License

Code: project authors. Beatmap assets: respective creators / osu! rights holders.
