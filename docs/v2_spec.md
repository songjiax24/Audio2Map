# Audio2Map v2 规格与实现状态

本文档是 **v2 管线的唯一权威参考**（不依赖对话 history）。  
旧 10ms sparse event MVP 见 [chart_tokens.md](chart_tokens.md) 末尾说明；`processed/` 为归档，新工作在 `processed_v2/`。

最后更新：2026-05-31（运行态见 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)）

---

## 0. 目标

```text
P(chart | audio, BPM, offset, condition_vector)
```

- **osu!mania 4K only**
- Phase 1：**constant BPM + constant meter + meter=4**
- 数据源：Sayobot Ranked/Approved（采集时已验证）
- **难度条件用 osu 官方 SR**，不是 OD；Etterna MSD 为 optional profile

---

## 1. 数据筛选（Phase 1）

| 条件 | 实现 |
|------|------|
| Mode = mania, 4K | `audio2map/data/chart_filter.py` |
| constant BPM | 所有 uninherited `beatLength` 相同 |
| constant meter | 所有 uninherited `meter` 相同 |
| meter == 4 | `REQUIRED_METER = 4` |

**统计脚本：** `scripts/analyze_dataset.py`  
必须报告：total / 4K / constant BPM / eligible / removed variable BPM / **removed meter≠4** / meter 分布 / BPM 分布 / hold / token 长度等。

eligible 规模（约）：~11.9k charts（全量 eligible，2026-05 扫描）。

---

## 2. offset 与 absolute tick（beat-grid reference）

**`offset_ms` 不是音乐开始、谱面开始或第一个 note 的时间。**  
它是 **beat grid 原点**：`absolute_tick = 0` 对应 `time_ms = offset_ms`。

```python
absolute_tick = round((time_ms - offset_ms) / tick_ms)
time_ms = offset_ms + absolute_tick * tick_ms
```

- 谱面事件可以出现在 offset **之前**：`time_ms < offset_ms` → **`absolute_tick < 0` 合法**
- **禁止** 丢弃 `tick < 0` 的事件；**禁止** 从 offset 裁切音频或谱面
- 内部 **`absolute_tick` 必须允许负数**

### 2.1 BPM canonicalization

```python
# audio2map/osu/bpm.py
def canonicalize_bpm(original_bpm) -> (canonical_bpm, bpm_scale_exp):
    # 乘/除 2 的幂，归到 [120, 240)
```

- **tokenization 与 audio tick 网格** 均用 `canonical_bpm` + `offset_ms`
- **不**改变音频播放速度；只改变 tick 离散化密度
- metadata 保存：`original_bpm`, `canonical_bpm`, `bpm_scale_exp`, `offset_ms`

```text
beat_ms = 60000 / canonical_bpm
tick_ms = beat_ms / 48
```

**代码：** `audio2map/osu/tick_range.py`, `audio2map/osu/row_tokens.py` (`ms_to_tick`, `split_tick`)

---

## 3. 时间网格与 bar 分解

```text
1 beat = 48 ticks
1 bar  = 4 beats = 192 ticks   (POS_0 .. POS_191)
```

**不用** beat/sub token；只用 `<BAR>` + `<POS_x>`。

对任意事件（`absolute_tick` 可为负）：

```python
absolute_bar = absolute_tick // 192          # Python floor division
pos = absolute_tick - absolute_bar * 192     # 0 <= pos < 192
```

例：`absolute_tick = -30` → `absolute_bar = -1`, `pos = 162` → token 窗口内 `<POS_162>`。

**Token 里的 `<BAR>` 是 window 内相对小节标记**，不编码 absolute bar 索引（可为负）。

训练窗口（debug 先固定）：

```yaml
window_bars_choices: [8]   # 后续 [8,12,16] 视 token 统计与显存
```

---

## 4. Tokenization（5-state ROW）

### 4.1 序列格式

```text
<BOS>
<ROW_initial>
<BAR>
<POS_x> <ROW_abcd>
...
<EOS>
```

- **空 bar 仍 emit `<BAR>`**
- 条件 **不进** token；见 §7 cond_vec

### 4.2 ROW 五态（L1–L4 = col0–col3）

| digit | 含义 |
|-------|------|
| 0 | empty |
| 1 | tap |
| 2 | hold_start |
| 3 | hold_active |
| 4 | hold_end |

词表：**625** ROW (`<ROW_0000>` … `<ROW_4444>`) + 192 POS + PAD/BOS/EOS/BAR → **vocab=821**

- **Initial ROW**：每位只能是 `{0,3}`，共 16 种；**prompt，不算 loss**
- **Event ROW**：至少含一个 `{1,2,4}`；纯 `{0,3}` 不作为事件 emit
- 有真实事件的 tick：其它 lane 若在 hold 中则标 **3**

### 4.3 Initial ROW（window 切点）

`<ROW_initial>` 表示 **window 起点** 时的 active hold 状态（prompt，loss=0）：

```python
window_start_tick = window_start_bar * 192
active_hold[lane] = any(hold.start_tick < window_start_tick < hold.end_tick)
state[lane] = 3 if active_hold[lane] else 0
```

`window_start_bar` **可为负**（offset 前弱起 / intro LN）。

### 4.4 Loss mask（continuous-window AR）

```text
loss = 0:  <BOS>, <ROW_initial>
loss = 1:  所有 <BAR>, <POS>, event <ROW>, <EOS>
```

无 ctx/tgt/future 边界 token；推理 overlap 是 **inference strategy**，不是训练 mask。

### 4.5 解码约束

- POS 在 BAR 内严格递增
- `active_hold[4]`；tap/start 仅 inactive lane；end 仅 active lane
- start→active，end→inactive

**代码：** `audio2map/osu/row_tokens.py`, `validate_token_sequence()`

---

## 5. Chart / window 范围

### 5.1 Chart event range（bar-aligned）

```python
first_event_tick = min(all tap ticks, all hold_start ticks, all hold_end ticks)
last_event_tick = max(...)

chart_start_bar = first_event_tick // 192
chart_end_bar = ceil((last_event_tick + 1) / 192)   # half-open [start, end)
```

`chart_start_bar` **可为负**。全曲 encode / hold_coverage 分母均基于此范围，**不是** 从 bar 0 起。

### 5.2 训练 window 采样

**不要** 假设 chart 从 bar 0 开始；**不要** 对空窗口做 rejection / max-empty-ratio。

```python
audio_start_bar = tick_min_audio // 192
audio_end_bar = ceil(tick_max_audio / 192)

train_start_bar = max(audio_start_bar, chart_start_bar - pre_event_margin_bars)
train_end_bar = min(audio_end_bar, chart_end_bar + post_event_margin_bars)

start_bar = randint(train_start_bar, train_end_bar - window_bars)
end_bar = start_bar + window_bars
```

推荐 margin（可调，优先于 empty-window rejection）：

```yaml
pre_event_margin_bars: 4
post_event_margin_bars: 4
```

空窗口在该范围内 **自然保留** — 谱面 break / 弱起前空段是真实分布，推理会覆盖整段 audio。

### 5.3 推理 window 范围

无 ground-truth events 时，按 **audio range** 覆盖整首歌：

```python
# 同上 audio_start_bar / audio_end_bar
# 生成窗口覆盖 [audio_start_bar, audio_end_bar)
```

后处理可裁掉 audio 范围外 notes 或异常 tail。

---

## 6. ms 量化 vs 原始 .osu（已知行为）

训练表示在 **tick 网格**上。原始 `time_ms` 经 round-trip 后：

- ~**56%** note 反量化 delta=0
- ~**44%** 有偏差；其中 **~94.5% 为 ±1ms**，最大观测 **±5ms**（与 BPM / round 有关）

**Round-trip 比较对象：** `quantize_notes(original)` ↔ `tokens_to_notes(tokens)`，不是原始 ms。  
**必须** 包含 `offset_ms > first_note_time_ms` 的 pre-offset 事件。

---

## 7. Condition vector（23 维，不进 token）

训练时 `cond_emb = MLP(cond_vec)` 注入 encoder + decoder。

| # | 字段 | 归一化 |
|---|------|--------|
| 1 | osu_sr_norm | clip(sr/10, 0, 1.5) |
| 2 | hold_ratio_norm | clip(hold_ratio, 0, 1) |
| 3 | hold_coverage_norm | clip(hold_coverage, 0, 1) |
| 4 | analyzer_ln_percent | [0,1] |
| 5 | analyzer_hb_row_ratio | [0,1] |
| 6–11 | analyzer Stream/Chordstream/Jacks/Coordination/Density/Wildcard | core_dist，和=1 |
| 12 | analyzer_available | 0/1 |
| 13–20 | etterna_*_norm (overall + 7 skillsets) | clip(msd/40, 0, 1.5) |
| 21 | msd_available | 0/1 |
| 22 | canonical_bpm_norm | (bpm-120)/120 |
| 23 | bpm_scale_exp_norm | scale_exp/4 |

**不可用规则：** analyzer/MSD 不可用时对应维度填 **0**，靠 `*_available` flag；不要把 0 当真实值。

**hold_ratio** = LN 数 / 总物件数  

**hold_coverage**（bar-aligned event range）：

```python
chart_total_ticks = (chart_end_bar - chart_start_bar) * 192
hold_coverage = total_hold_lane_ticks / (chart_total_ticks * 4)
# chart_total_ticks <= 0 → hold_coverage = 0
```

**不在 cond_vec、但切片需要：** `offset_ms`（audio window 对齐）

**实现：** `audio2map/data/cond_vec.py` — `build_cond_vec(ChartMeta) -> np.ndarray[23]`

---

## 8. Audio tick-grid

- 22.05kHz，142 维/tick：mel128 + onset + rms + chroma
- **离线** pooled tick-grid cache → `processed_v2/audio_grid/`

**实现：** `audio2map/audio/tick_features.py`, `audio2map/data/audio_grid.py`, `scripts/precompute_audio_grid.py`

### 8.1 Audio 覆盖的 absolute tick range

音频文件开头可能在 offset 之前，grid **不能** 默认从 tick 0 起：

```python
tick_min_audio = floor((0.0 - offset_ms) / tick_ms)
tick_max_audio = ceil((audio_duration_ms - offset_ms) / tick_ms)
# half-open [tick_min_audio, tick_max_audio)
```

`tick_min_audio` 通常为 **负数**。

### 8.2 数组索引与切片

```python
array_index = absolute_tick - tick_min_audio
absolute_tick = array_index + tick_min_audio

window_start_tick = window_start_bar * 192
window_end_tick = window_end_bar * 192
start_idx = window_start_tick - tick_min_audio
end_idx = window_end_tick - tick_min_audio
# 超出 audio grid → padding
```

缓存 metadata 示例：

```json
{
  "tick_min": -123,
  "tick_max": 25920,
  "offset_ms": 1234.0,
  "canonical_bpm": 180.0,
  "ticks_per_beat": 48
}
```

### 8.3 Cache key

tick-grid 依赖 `audio_hash + canonical_bpm + offset_ms`：

```text
<audio_hash>_bpm180.000_offset1234.0.npy
```

**代码：** `audio2map/osu/tick_range.py` (`audio_grid_cache_stem`, `audio_tick_range_ms`)

---

## 9. 训练样本结构（示意）

```json
{
  "set_id": 871715,
  "beatmap_id": 1867061,
  "window": {"start_bar": -2, "window_bars": 8},
  "tokens": ["<BOS>", "<ROW_0003>", "<BAR>", "..."],
  "loss_mask": [0, 0, 1, 1, ...],
  "cond_vec": [23 floats],
  "audio_slice": {"start_tick": -384, "length_ticks": 1536, "features": 142}
}
```

`start_bar` 可为负；token 内 `<BAR>` 仍为 window-local。

---

## 10. 推理（计划）

Overlap generation：`context_bars + keep_bars + future_bars = window_bars`  
生成全窗，只保留 keep 区，丢弃 future；prompt 可含 `<BOS><ROW_initial>` + 已生成 context tokens。

**覆盖范围：** 整首 `[audio_start_bar, audio_end_bar)`（§5.3），与训练采样范围不同。

---

## 11. 目录布局

```text
/root/audio2map/                 # 代码
/root/autodl-tmp/audio2map_data/
├── raw/{sid}/                   # mp3 + *.osu
├── processed/                   # 旧 MVP（可删）
├── processed_v2/                # v2 运行时产物
│   ├── audio_grid/            # {stem}.npy + .json
│   ├── checkpoints/           # e.g. trial_multi/, abendstern_overfit/
│   └── logs/
├── chart_meta/manifest.jsonl
└── .collector/
```

---

## 12. 代码索引

**新 session / 克隆实例：** 先读 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)（checkpoint、precompute、命令、已知 bug）。

| 模块 | 路径 |
|------|------|
| tick / bar range | `audio2map/osu/tick_range.py` |
| cond_vec | `audio2map/data/cond_vec.py` |
| audio_grid | `audio2map/data/audio_grid.py` |
| window sampling | `audio2map/data/window_sampler.py` |
| training sample builder | `audio2map/data/v2_dataset.py` |
| constrained decode | `audio2map/training/decode.py` |
| overlap inference | `audio2map/training/inference.py` |
| export .osu | `audio2map/osu/export.py` |
| train / infer scripts | `scripts/train_v2.py`, `scripts/infer_v2.py` |
| 5-state ROW + vocab | `audio2map/osu/row_tokens.py` |
| Round-trip | `audio2map/osu/round_trip.py` |
| BPM canon | `audio2map/osu/bpm.py` |
| Chart filter | `audio2map/data/chart_filter.py` |
| hold_ratio / hold_coverage | `audio2map/data/chart_stats.py` |
| Pattern API | `audio2map/pattern_analyser/analyze.py` |
| Chart meta batch | `scripts/compute_chart_meta.py` |
| Dataset stats | `scripts/analyze_dataset.py` |

---

## 13. 实现顺序

1. ✅ v2 vocab + 5-state ROW  
2. ✅ negative tick + bar-aligned range（encode / quantize / round-trip）  
3. ✅ canonical BPM + filter + analyze_dataset（部分）  
4. ✅ pattern analyser vendored  
5. ✅ hold_coverage bar-aligned 公式  
6. ✅ `build_cond_vec()` — `audio2map/data/cond_vec.py`  
7. ✅ audio_grid 预计算（tick_min/max + cache key）— `scripts/precompute_audio_grid.py`  
8. ✅ Dataset 样本构建 + window 采样 — `audio2map/data/v2_dataset.py`  
9. ✅ PyTorch DataLoader + debug overfit — `scripts/train_debug_overfit.py`  
10. ✅ constrained decoding + overlap inference — `audio2map/training/decode.py`, `inference.py`  
11. ✅ export `.osu` — `audio2map/osu/export.py`, `scripts/infer_v2.py`  
12. ✅ eval metrics — `audio2map/eval/`, `scripts/eval_v2.py`  
13. 🟡 multi-chart trial train done (1109 charts, 3k steps); precompute **67%** then **disk full** — see [AGENT_HANDOFF.md](AGENT_HANDOFF.md)  

---

## 14. 测试要求（round-trip / analyzer）

必须包含 **pre-offset** 样例（`offset_ms > first_note_time_ms`）：

```text
.osu → absolute_tick (可负) → BAR/POS tokens → .osu
```

确认：

- negative `absolute_bar` / `window_start_bar`
- `audio_grid` 的 `tick_min < 0`
- negative `window_start_tick` 上的 `<ROW_initial>`（intro LN）
- 导出 ms 误差在 tick 量化误差内

**示例 set_id：** 2127527 (Abendstern Insane), 2545208, 2306091

**单元测试：** `tests/test_row_tokens_v2.py` — `test_pre_offset_events_round_trip`, `test_split_negative_tick`

---

## 15. 数据集统计备忘（pre-offset，2026-05）

全量 eligible（11927 charts）中约 **17.4%** 含 head tick&lt;0；此前错误实现静默丢弃 **10694** note heads。修复后应全部保留。

---

## 16. 用户确认记录（摘要）

- Phase 1：meter=4 only；空 bar 仍 `<BAR>`  
- canonical BPM 只改 tick 密度，不改音频速度  
- offset = beat-grid origin；**不裁** audio/chart；tick 可负  
- hold_coverage 分母：bar-aligned chart event range  
- initial ROW 16 种；event ROW 625 中仅含 {1,2,4} 的 609 种可预测  
- 训练采样：event range + margin，clamp 到 audio；**无** empty-window rejection  
- 推理：整首 audio range  
- Pattern：vendor Python port；官方 SR + MSD 离线 batch  
- 训练：continuous-window AR；cond 不进 token  

---

## 17. 环境

```bash
conda activate audio2map
export AUDIO2MAP_DATA_ROOT=/root/autodl-tmp/audio2map_data
```

依赖：`rosu-pp-py`, vendored minacalc runner, pattern_analyser（无 nonebot）。
