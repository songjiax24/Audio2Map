# Agent handoff — Audio2Map v2

> **⚠️ 本页为 Agent 工作笔记，未经 sjx 确认，可能过时。**  
> **唯一可信规格：[OVERVIEW.md](OVERVIEW.md)**。不确定先问 sjx。

| 文档 | 状态 | 用途 |
|------|------|------|
| **[OVERVIEW.md](OVERVIEW.md)** | **sjx 确认** | 方案概要（唯一可信） |
| [V2_MASTER_SPEC.md](V2_MASTER_SPEC.md) | unverified | Agent 整理的“主规格” |
| [V2_COMPLIANCE_AUDIT.md](V2_COMPLIANCE_AUDIT.md) | unverified | 代码 vs Agent 文档对照 |
| [v2_spec.md](v2_spec.md) | unverified | 数据管线、代码索引 |
| [../README.md](../README.md) | unverified | 命令速查 |

**规则：不确定处先问 sjx；Agent 不得自行改 OVERVIEW 或做 breaking 决定。**

Last updated: **2026-06-01**

---

## 0. Quick start (5090)

```bash
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
cd /root/audio2map && source .venv/bin/activate

python scripts/audit_audio_grid.py      # expect valid=3567, invalid=0
pytest tests/test_v2_pipeline.py -q

# Formal training (defaults: 11927 charts, enc_dec, 50k steps → train_v1/)
# ⚠️ Do NOT run until V2_COMPLIANCE_AUDIT.md §27 gates pass
python scripts/train_v2.py --architecture enc_dec
```

**Do not clone the old system disk** — only mount the data disk and `git clone` this repo.

---

## 1. Goal

```text
P(chart | audio, BPM, offset, condition_vector)
```

- osu!mania **4K**, Phase 1: constant BPM, meter=4
- `offset_ms` = beat-grid origin; **negative ticks allowed**, no cropping
- **Generative** task: one training chart is one sample, not the unique answer
- `token_acc` in logs = teacher-forcing diagnostic only (not “higher is better” for final chart quality)

---

## 2. Paths

| What | Path |
|------|------|
| Code | `/root/audio2map` |
| Data root | `/root/autodl-tmp/audio2map_data` |
| Raw sets | `$DATA/raw/{set_id}/` |
| Audio grids | `$DATA/processed_v2/audio_grid/{stem}.npy` + `.json` |
| Checkpoints (formal) | `$DATA/processed_v2/checkpoints/train_v1/` |
| Checkpoints (legacy trial) | `$DATA/processed_v2/checkpoints/trial_multi_v2/` |
| Logs | `$DATA/processed_v2/logs/` |
| Generated exports | `$DATA/processed_v2/generated/` |
| Chart meta | `$DATA/chart_meta/manifest.jsonl` |

Override: `export AUDIO2MAP_DATA_ROOT=...`

**Safe to delete on disk:** `$DATA/processed/` (~10G) — deprecated v1 mel cache.

---

## 3. Environment

PyTorch **2.12+cu130**, RTX 5090, uv venv 3.11. See [README.md](../README.md) for setup.

---

## 4. Current state (2026-06-01)

### Data / precompute — **COMPLETE**

| Item | Status |
|------|--------|
| Eligible charts | **11,927** |
| Valid audio grids | **3,567** (audit: **0 invalid**) |
| Trainable (`--require-grid` default) | **11,927 / 11,927** (100%) |
| Hold-out (no grid) | **0** |

Precompute 已跑完；正式训练不再依赖续跑 precompute。

### Model / training architecture (formal)

| Item | Formal default | Legacy trial |
|------|----------------|--------------|
| `audio_pooling` | **`tick`** (1536 ticks/window) | `bar` (8 vectors/window) |
| Charts | **11,927** | 1,109–6,159 (partial grids) |
| Checkpoint dir | **`train_v1/`** | `trial_multi_v2/` |
| Steps default | **50,000** | 3,000–20,000 |

**Legacy `trial_multi_v2/step_20000.pt`:** bar-pool 模型，6159 张谱时代产物；**不能**作为正式训练目标，仅作历史对照。

### Code fixes already in tree (2026-06-01)

- First overlap window: `context_bars=0`（不丢首 4 小节）
- Inference: 默认 `audio_full` range + tail 分析指标；eval 可用 `reference_chart`
- Decode: 默认采样 `T=0.8, top_p=0.95`（`--greedy` 仅 debug）
- Degeneracy metrics + split token accuracy

### Next: formal training run

```bash
python scripts/train_v2.py \
  --out "$AUDIO2MAP_DATA_ROOT/processed_v2/checkpoints/train_v1" \
  2>&1 | tee "$AUDIO2MAP_DATA_ROOT/processed_v2/logs/train_v1.log"
```

Preload ~11k charts 需 **~1–2 小时**；之后 ~25 step/s。无 `--resume`，长跑请自行规划 save_every。

---

## 5. Commands

```bash
# Grids (maintenance only — precompute done)
python scripts/audit_audio_grid.py

# Formal train (defaults: require_grid, tick pooling, 50k steps)
python scripts/train_v2.py

# Single-window debug generate
python scripts/infer_v2.py --checkpoint .../step_50000.pt --osu "..." \
  --single-window --start-bar -1

# Split teacher accuracy
python scripts/eval_v2.py --mode teacher --checkpoint ... --osu "..." --start-bar -1

# ROW / hold stats (no loss changes)
python scripts/analyze_row_token_stats.py --limit 500
```

---

## 6. Training mechanics

| Term | Meaning |
|------|---------|
| **step** | One batch (8 windows) → one optimizer update |
| **preload** | Load all chart bundles into RAM before step loop |
| **samples/epoch** | `11927 × samples_per_chart` (default 23,854) |
| **encoder prefix** | tick: **1537** tokens (1 cond + 1536 ticks) per 8-bar window |

Model: **~3.7M params**, vocab **821**, window **8 bars**, cond_vec **23-d**.

---

## 7. Known bugs fixed (do not revert)

See [AUDIO_TEMPORAL_AUDIT.md](AUDIO_TEMPORAL_AUDIT.md), inference first-window fix, `is_valid_audio_grid_cache()`, etc.

---

## 8. Related docs

- [v2_spec.md](v2_spec.md)
- [AUDIO_TEMPORAL_AUDIT.md](AUDIO_TEMPORAL_AUDIT.md)
- [PROJECT_CONVERSATION_LOG.md](PROJECT_CONVERSATION_LOG.md)
- [../configs/README.md](../configs/README.md)
