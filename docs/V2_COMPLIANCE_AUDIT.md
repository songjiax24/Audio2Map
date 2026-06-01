# V2 主规格合规审计

> **⚠️ 未经 sjx 确认。** Agent 对照 [V2_MASTER_SPEC.md](V2_MASTER_SPEC.md)（本身亦未确认）与代码写的审计。  
> **合规基准应是 [OVERVIEW.md](OVERVIEW.md)**；本文件仅作历史参考，不可当作验收标准。

审计日期：2026-06-01

---

## 1. 十条原则对照

| # | 原则 | 当前状态 | 说明 |
|---|------|----------|------|
| 1 | offset = beat origin；tick 可为负 | ✅ 基本符合 | `tick_range.py`, `row_tokens.py`；export 过滤 `time_ms<0` 是 osu 兼容后处理 |
| 2 | audio tick-level；禁止 bar-pool | ⚠️ 部分 | `enc_dec` 已 tick-level；`prefix_lm` ablation 仍存在且可 `--architecture prefix_lm` |
| 3 | BOS + initial ROW + BAR/POS/ROW + EOS | ✅ 符合 | `row_tokens.py` |
| 4 | initial ROW 是 prompt | ✅ 符合 | `build_loss_mask`: BOS + ROW_initial loss=0 |
| 5 | cond_emb 注入 encoder/decoder **所有** token | ⚠️ 部分 | `enc_dec` 已 broadcast；**缺 LayerNorm**；`cond_vec` **维度顺序与主规格 §14.1 不一致** |
| 6 | 主模型 = encoder-decoder | ⚠️ 部分 | `AudioChartModel` 已实现；**默认 train 仍可选 prefix_lm**；旧 smoke ckpt 是 prefix_lm |
| 7 | audio encoder 非因果 | ✅ enc_dec 符合 | `TransformerEncoder` 无 causal mask |
| 8 | decoder cross-attend audio memory | ✅ enc_dec 符合 | `TransformerDecoder` |
| 9 | continuous-window AR | ✅ 符合 | 整 window loss（除 BOS/initial ROW） |
| 10 | overlap 是推理策略 | ✅ 符合 | 未写入训练 loss |

**结论：主架构方向已改 enc_dec，但仍有多处与主规格 **未对齐**；**不得开始 formal 11.9k 训练**，直到 §27 sanity gates 通过。**

---

## 2. §29 必须纠正项 — 当前代码状态

| 项 | 要求 | 当前 | 优先级 |
|----|------|------|--------|
| prefix-LM 主模型 | 废弃主线 | `PrefixLMAudioChartModel` 仍可通过 CLI 训练 | P0 禁用默认/正式路径 |
| cond 单 prefix token | 禁止 | enc_dec 已改 broadcast | ✅ |
| audio 无 position emb | 禁止 | enc_dec 已有 4 种 time emb | ✅ |
| audio causal | 禁止 | enc_dec 已非因果 | ✅ |
| bar-level pooling | 禁止 | 仅 prefix_lm ablation | P0 正式训练禁用 prefix_lm |
| window_bars=8 写死 | 须 `window_bars_choices` | 机制已有；**默认仍 `(8,)`** | P1 debug `[8]` OK；formal 待统计后 `[8,12,16]` |
| 仅 token_acc | 须拆分指标 | 部分在 `eval/`；训练 log 仍 mainly token_acc | P2 |
| greedy 默认 | 禁止 | infer 默认 sampling ✅ | ✅ |

---

## 3. 与主规格仍不一致的实现细节

### 3.1 cond_vec 维度顺序（§14.1）

**主规格顺序：** analyzer 3–10 → etterna 11–18 → bpm 19–20 → flags 21–22

**当前 `cond_vec.py`：** analyzer 3–10 → **analyzer_available@11** → msd 12–19 → **msd_available@20** → bpm 21–22

→ **与主规格不一致，需迁移（breaking change）。**

### 3.2 cond_mlp（§18.4）

主规格：`Linear → GELU → Linear → LayerNorm`

当前：`Linear → GELU → Linear`（**无 LayerNorm**）

### 3.3 模型深度默认值（§18.5）

| | debug | formal |
|---|-------|--------|
| 主规格 encoder/decoder | 2 / 2 | 4 / **6** |
| 当前 `train_v2` 默认 | 4 / 4 | 4 / 4 |

### 3.4 Audio feature（§7）— **今晚策略：沿用磁盘 v1，不 precompute**

| 项 | 主规格目标 (v2) | **当前代码 + 3567 grids (v1)** |
|----|-----------------|----------------------------------|
| hop | 512 固定 | **动态 `tick_ms/4`** |
| pooling | tick-centered | **frame floor 分桶 + mean** |
| onset | max | **mean** |
| cache dtype | float16 | **float32** |

→ v2 feature 留到小样本对比后再 selective precompute。**训练/推理读现有 grid 与 v1 代码一致。**

### 3.5 Sanity gates（§27）

| Gate | 状态 |
|------|------|
| Round-trip | ✅ tests 有；需含 pre-offset 等 edge cases |
| Single-map overfit | ❌ **enc_dec 上未跑通门禁** |
| Audio ablation | ❌ 未系统化 |
| Position emb ablation | ❌ 未做 |

### 3.6 其他

- **condition dropout**（§14.3）：未实现（第一版可关，正式训练再开）
- **max_token_len 统计**（§22）：未跑
- **train_v1_smoke ckpt**：prefix_lm，**不可用于 enc_dec 评估**

---

## 4. 改动计划（按 §28 实施顺序，未获确认不动 breaking 项）

### Phase A — 文档与门禁（无 breaking）

1. ✅ 写入 `V2_MASTER_SPEC.md`、`V2_COMPLIANCE_AUDIT.md`
2. 更新 `AGENT_HANDOFF.md` / `v2_spec.md` 指向主规格
3. `train_v2.py`：**正式训练仅允许 `enc_dec`**（prefix_lm 需显式 `--ablation-prefix-lm`）
4. 添加 CI/pretrain checklist 脚本占位（overfit / ablation 入口）

### Phase B — 模型对齐（小 diff）

5. `cond_proj` 末尾加 `LayerNorm`
6. 拆分 `encoder_layers` / `decoder_layers` CLI；formal preset = 4/6
7. 确认 decoder 使用 `decoder_abs_pos_emb[i]`（当前 `decoder_pos` 语义一致，仅 rename/doc）

### Phase C — cond_vec 迁移（**breaking，需 sjx 确认**）

8. 按 §14.1 重排 `COND_VEC_NAMES` + `build_cond_vec`
9. 重算/迁移 `chart_meta/manifest.jsonl` 若已缓存 cond 相关字段
10. 旧 checkpoint **全部作废**（已预期）

### Phase D — Audio feature v2（**延后**，小样本对比后再定）

11. ~~重写 tick_features~~ → 当前 **v1 与磁盘 grid 对齐**；v2 对比实验另开分支/脚本
12. 全量 precompute：**不在今晚 / 未对比前执行**

### Phase E — Sanity gates（blocking formal train）

14. `train_debug_overfit.py` on enc_dec → loss≈0，free gen 复现 LN
15. Audio ablation 脚本（normal/zero/shuffle）
16. Position emb ablation 开关
17. 拆分训练/评估指标（§26）

### Phase F — Formal training

18. 仅当 E 全部通过 → `11.9k × 50k+`，`window_bars_choices [8,12,16]`（若 §22 统计允许）

---

## 5. 待 sjx 确认（⚠️ 未确认前不做）

1. **cond_vec 维度重排**：是否立即按 §14.1 改？现有 manifest/训练样本是否全部 invalidate？
2. **Audio feature v2**：延后；当前 v1 与 3567 grids 兼容 — **已对齐**
3. **prefix_lm 代码**：完全删除 vs 保留 `--ablation-prefix-lm` 仅本地对照？
4. **Formal 默认层数**：debug 2/2、formal 4/6 是否作为 `train_v2.py` 的 `--preset debug|formal`？
5. **float16 grid**：训练时 cast float32 还是 loader 内 upcast？

---

## 6. 明确不做（除非 sjx 改 spec）

- 不把 prefix_lm 继续当主线训练/汇报
- 不在 audio ablation 未通过时扩 formal 数据实验
- 不在不确定 cond/audio breaking change 时自行迁移 checkpoint
- 不用 bar/beat pooling 换 tick-level 省显存
