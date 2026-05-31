# Agent handoff — Audio2Map v2

**Read this first** when resuming work on a fresh or cloned instance.  
Authoritative model/data spec: [v2_spec.md](v2_spec.md).

Last updated: **2026-05-31**

---

## 0. Migration note (5090 instance)

**Do not clone the old system disk.** Only attach/mount the **data disk**:

```text
/root/autodl-tmp/audio2map_data/   # raw, processed_v2, chart_meta, …
```

On the new machine:

1. `git clone` / `git pull` this repo to `/root/audio2map` (or any path)
2. Recreate Python env (conda **or** uv — see §3)
3. `export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data`
4. Install **PyTorch with CUDA build matching the 5090 driver** (see §3.2)
5. Read §4 for checkpoint / precompute state on the data disk

Checkpoints and `audio_grid/` caches are **on the data disk**, not in git.

---

## 1. Goal

```text
P(chart | audio, BPM, offset, condition_vector)
```

- osu!mania **4K**, Phase 1: **constant BPM**, **meter=4**
- `offset_ms` = beat-grid origin (`absolute_tick=0` at `offset_ms`); ticks **may be negative**; **no cropping**
- Training: continuous-window AR; cond_vec is **not** tokenized (prefix conditioning)

---

## 2. Paths (default AutoDL layout)

| What | Path |
|------|------|
| Code repo | `/root/audio2map` |
| Data root | `/root/autodl-tmp/audio2map_data` |
| Raw sets | `$DATA/raw/{set_id}/` — audio + `*.osu` |
| v2 artifacts | `$DATA/processed_v2/` |
| Audio grids | `$DATA/processed_v2/audio_grid/{stem}.npy` + `.json` |
| Checkpoints | `$DATA/processed_v2/checkpoints/` |
| Logs | `$DATA/processed_v2/logs/` |
| Chart meta | `$DATA/chart_meta/manifest.jsonl` |

```bash
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
conda activate audio2map
cd /root/audio2map
```

Override data root via `AUDIO2MAP_DATA_ROOT` (see `audio2map/utils/paths.py`).

---

## 3. Environment

### 3.1 Recommended: `uv` (new instances)

**Yes — uv is a good fit** for this repo on a fresh 5090 box:

- Fast, reproducible venv from `pyproject.toml`
- No need to clone the old conda env
- Keep **ffmpeg** as a system/apt package (librosa decode)

```bash
# system (AutoDL / Ubuntu)
apt-get update && apt-get install -y ffmpeg libsndfile1   # if missing

curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/audio2map

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,train]"

# PyTorch: pick the wheel for YOUR driver / GPU (5090 = very new CUDA arch)
# Check https://pytorch.org/get-started/locally/ — often cu124/cu128 or nightly.
# Example (adjust index URL to match PyTorch site):
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest tests/test_row_tokens_v2.py tests/test_v2_pipeline.py -q
```

If `torch.cuda.is_available()` is false or you get SM/arch errors, reinstall torch from a **newer** CUDA wheel — do not assume the old 4090 env transfers.

### 3.2 Alternative: conda (legacy)

```bash
conda env create -f environment.yml
conda activate audio2map
# Still reinstall torch on 5090 if CUDA mismatch:
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 3.3 Runtime

```bash
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
```

- GPU training tested on RTX **4090** (old instance); **5090 needs fresh torch**
- `precompute_audio_grid.py` is **CPU-heavy** (librosa); low GPU util during preload is normal

---

## 4. Session state (2026-05-31)

### Done / verified

| Item | Result |
|------|--------|
| v2 tokenization + negative tick round-trip | 500-chart sample **100%** |
| Abendstern single-window overfit | token_acc **1.0** (`checkpoints/abendstern_overfit/final.pt`) |
| Teacher eval (same window) | token_acc 1.0, note F1 1.0 |
| Generate AR (overfit ckpt, fixed `next_token_logits`) | note F1 1.0 (single window only) |
| Multi-chart trial train | **3000 steps**, 1109 charts, see below |
| Eval pipeline | `scripts/eval_v2.py` — roundtrip / teacher / generate / window |

### Multi-chart trial (`trial_multi`)

- **Completed** 3000 steps (~19 step/s after preload)
- **1109** charts with precomputed `audio_grid` → **2218** samples/epoch
- Checkpoints: `processed_v2/checkpoints/trial_multi/step_{500,2000,2500,3000}.pt`
- Final log metrics: loss ≈ 3.86, token_acc ≈ **31%** (not converged; peak acc ~50% mid-run)
- Log: `processed_v2/logs/trial_multi_train.log`

### Precompute (`audio_grid`)

- Total unique jobs: **3564** sets (dedupe by `audio_hash + canonical_bpm + offset_ms`)
- Progress when stopped: **~2402/3564 (67%)**
- **Stopped intentionally** — data disk **50G/50G full** (`Errno 28 No space left on device`)
- Log: `processed_v2/logs/precompute_audio_grid.log` (~684 disk-full warnings at tail)
- **Do not restart precompute** on 50G disk without expanding storage or freeing space

**Disk budget rule of thumb:** each grid ≈ **15–25 MB** (full song, 142-dim float32 per tick).  
3564 sets ≈ **45–70 GB** for grids alone, plus `raw/` audio. Plan **≥80–100 GB** data disk for full precompute.

### Known bugs fixed in code (must stay)

1. **`AudioChartModel.next_token_logits()`** — generation must append PAD before `forward()`; do not use `forward()` alone for single-step decode.
2. **`max_seq_len=1024`** in `model.py` — 8-bar windows can reach **~559 tokens**; old limit 512 caused CUDA index error in multi-chart training.
3. **`filter_paths_with_grid()`** in `chart_bundle.py` — used by `train_v2.py --require-grid` to skip charts without cache (faster startup).
4. **`ChartBundle` preload** — `scripts/train_debug_overfit.py` preloads one batch; avoid recomputing SR/grid every step.

---

## 5. Commands cheat sheet

```bash
# Precompute (CPU, long; needs disk space)
python scripts/precompute_audio_grid.py
python scripts/precompute_audio_grid.py --limit 100   # smoke

# Train
python scripts/train_v2.py --require-grid --steps 3000 --batch-size 8 \
  --samples-per-chart 2 --out processed_v2/checkpoints/trial_multi

# Single-chart overfit debug
python scripts/train_debug_overfit.py \
  --osu "/root/autodl-tmp/audio2map_data/raw/2127527/seatrus - Abendstern (Maiiy) [Insane].osu" \
  --start-bar -1 --steps 1200

# Eval
python scripts/eval_v2.py --mode teacher --checkpoint .../final.pt \
  --osu "..." --start-bar -1
python scripts/eval_v2.py --mode generate --checkpoint ... --osu "..." --start-bar -1

# Tests
pytest tests/test_row_tokens_v2.py tests/test_v2_pipeline.py tests/test_eval.py tests/test_decode_inference.py -q
```

Config YAML (reference only; scripts use CLI): `configs/data/v2.yaml`, `configs/train/trial_multi.yaml`.

---

## 6. Key modules

| Concern | Module |
|---------|--------|
| Spec / tokens | `audio2map/osu/row_tokens.py`, `docs/v2_spec.md` |
| cond_vec (23-d) | `audio2map/data/cond_vec.py` |
| Tick audio features | `audio2map/audio/tick_features.py` (22.05 kHz, 142-d/tick) |
| Grid cache | `audio2map/data/audio_grid.py` |
| Window sampling | `audio2map/data/window_sampler.py` |
| Training sample | `audio2map/data/v2_dataset.py` |
| Fast preload | `audio2map/data/chart_bundle.py` |
| Model | `audio2map/training/model.py` — `AudioChartModel` |
| Dataset / collate | `audio2map/training/dataset.py`, `collate.py` |
| Constrained decode | `audio2map/training/decode.py` |
| Overlap inference | `audio2map/training/inference.py` |
| Export .osu | `audio2map/osu/export.py` |
| Eval | `audio2map/eval/chart_eval.py`, `note_match.py` |

Vocab size: **821** = 4 specials + 192 POS + 625 ROW.

---

## 7. Benchmark charts

| set_id | Use |
|--------|-----|
| **2127527** | Abendstern Insane — pre-offset events, overfit/eval baseline |
| 2545208, 2306091 | Additional pre-offset round-trip tests |

Example path:

```text
/root/autodl-tmp/audio2map_data/raw/2127527/seatrus - Abendstern (Maiiy) [Insane].osu
```

Eligible charts: ~**11.9k**; ~**17.4%** have head `tick < 0`.

---

## 8. Recommended next steps

1. **New instance:** expand data disk (≥80–100 GB) or accept partial grid cache (~1109+ charts trainable today).
2. **Stop zombie precompute** if still running: `pkill -f precompute_audio_grid`
3. **Eval** `trial_multi/step_3000.pt` on random windows (not overfit start_bar).
4. **Longer training** (20k–50k steps) or tune LR; current 31% acc is early.
5. **Precompute resume** only after disk fix; script skips existing `.npy` by default.
6. Optional optimizations: fp16 grids, manifest cache for cond_vec preload, `num_workers>0` DataLoader.

---

## 9. What is NOT in git

Data and checkpoints live on the **data disk**, not in `/root/audio2map`:

- `autodl-tmp/audio2map_data/raw/`
- `autodl-tmp/audio2map_data/processed_v2/`

**Migration plan:** mount/copy only the data disk; `git clone` the repo separately on the new instance.  
Set `AUDIO2MAP_DATA_ROOT` to wherever the data disk is mounted.

---

## 10. Precompute stop / disk check

```bash
df -h /root/autodl-tmp
pgrep -af precompute_audio_grid || echo "not running"
pkill -f precompute_audio_grid   # if needed
du -sh /root/autodl-tmp/audio2map_data/*   # find large dirs
ls processed_v2/audio_grid/*.npy | wc -l   # grid count
```

---

## 11. Related docs

- [v2_spec.md](v2_spec.md) — full spec + implementation checklist
- [chart_tokens.md](chart_tokens.md) — legacy 10 ms MVP
- [data_collection.md](data_collection.md) — Sayobot collector
