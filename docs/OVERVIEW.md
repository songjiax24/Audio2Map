# osu!mania 4K 谱面生成方案概要

> **sjx 确认的方案概要 — 仓库内唯一可信规格。**  
> 其余文档（`V2_MASTER_SPEC.md`、`v2_spec.md`、`AGENT_HANDOFF.md`、合规审计、代码注释等）均为 Agent 整理或实现产物，**未经 sjx 逐条确认**，可能与本文或实际代码不一致。  
> **冲突时以本文为准；不确定处先问 sjx，Agent 不得自行拍板或改 spec。**

> **【审阅说明】** 各节末尾的 **【实现注记】** 为 Agent 对照当前代码仓库写的对照表（2026-06-01），标注 ✅ 基本一致 / ⚠️ 部分一致或有细节差异 / ❌ 未实现或明显偏离。**注记本身也未经 sjx 确认**，供你审阅时跳转代码用。  
> **仓库内每一 Python 文件 / 脚本 / 测试 / 配置** 均在 **[附录 A：代码全量索引](#附录-a仓库代码全量索引)** 中列出；各节注记侧重该节相关代码，附录保证无遗漏。

我们的初始目标是建模条件分布：

```text
P(chart | audio, cond_vec)
```

也就是说，给定一首音乐的音频信息，以及一组描述目标谱面风格与难度的条件向量 `cond_vec`，模型需要生成一张与音乐对齐、满足条件约束、且具有合理键形结构的 osu!mania 4K 谱面。

这里的 `chart` 指 osu!mania 4K 谱面中的 hit objects，包括普通 tap 和长条 hold/LN。`audio` 是输入音乐。`cond_vec` 是全局条件向量，包含预期难度、长条比例、键型比例/难度、BPM 相关信息等。

> **【实现注记】** *(§0 总览)*
>
> | | |
> |---|---|
> | **目标函数** | ✅ 训练/推理管线均围绕 chart token 序列 + 音频 + cond_vec；入口见 `scripts/train_v2.py`、`scripts/infer_v2.py` |
> | **BPM/offset** | ⚠️ OVERVIEW 此处写 `P(chart \| audio, cond_vec)`；代码里 BPM/offset 通过 `CanonicalTiming` 进入 tick 网格与 cond_vec（`canonical_bpm_norm`），未作为独立张量输入模型 |
> | **主链路** | `data/chart_bundle.py` → `training/dataset.py` → `training/model.py` → `training/inference.py` → `osu/round_trip.py` → `osu/export.py` |

---

## 1. 任务限定

为了降低问题复杂度，并使模型能够优先学习 osu!mania 4K 中最核心的节奏、键形和长条规律，我们将第一阶段任务限定为：

```text
- osu!mania
- 4 keys
- constant BPM
- meter = 4
```

也就是说，第一阶段暂不处理：

```text
- 变 BPM
- 非 4/4 拍
- 变速 / SV
- hitsound
- kiai
- 多 key count
- 非 osu!mania 模式
```

这样做牺牲了一部分通用性，但可以显著简化时间表示和模型学习目标。尤其是 osu!mania 谱面本身高度依赖节拍网格和 snap 位置，与其使用毫秒级 time grid，不如直接在 beat grid 上建模谱面事件。这也更贴近人类制谱时基于拍点、分拍和小节结构安排键形的过程。

> **【实现注记】** *(§1 任务限定)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | mania 4K | ✅ | `osu/mania.py` — `is_mania_4k_sections()`；解析时校验 CircleSize=4 |
> | constant BPM | ✅ | `data/chart_filter.py:64-65` — `timing.constant_bpm` |
> | constant meter | ✅ | `data/chart_filter.py:66-67` |
> | meter = 4 | ✅ | `data/chart_filter.py:68-69`，`REQUIRED_METER = 4` |
> | 变 BPM / 非 4/4 | ✅ 排除 | 不合格谱写入 `FilterReason`；训练集 `data/eligible_charts.py` → `list_eligible_osu_paths()` |
> | hitsound / kiai / SV | ⚠️ 未进 token | 解析器不把它们编进序列；**采集阶段也未显式拒绝**含这些字段的谱 |
> | 非 mania / 非 4K | ✅ 排除 | `FilterReason.NOT_4K` |
>
> **入口脚本：** `scripts/collect.py`（采集）、`scripts/analyze_dataset.py`（统计合格率）、`scripts/analyze_timing.py`（BPM 分布）  
> **采集实现：** `data/collector/sayobot.py`、`cli.py`、`state.py`  
> **测试：** `tests/test_row_tokens_v2.py`（`check_parsed_beatmap`）、`tests/test_osu_parser.py`  
> **遗留代码：** `audio2map/legacy/`（`events.py`、`tokens.py`、`frames.py`）— v1 10 ms 稀疏事件，**不参与 v2 训练**  
> **其余文件：** 见 [附录 A.4、A.11](#a4-audio2mapdata--数据集缓存采集)

---

## 2. Beat grid 表示

在生成谱面前，我们首先测定或读取音乐的：

```text
BPM
offset
```

其中 `offset` 表示 beat grid 的参考原点，而不是音乐文件的开始时间。因此，谱面事件可能出现在 offset 之前，对应的 tick 可以为负。

对于 constant BPM 且 meter = 4 的音乐，我们构建如下节拍网格：

```text
1 beat = 48 ticks
1 bar = 4 beats = 192 ticks
```

即：

```text
1 tick = 1/48 beat
```

选择 1/48 beat 是因为它能够覆盖 osu!mania 谱面中常见的大多数 snap 粒度，例如：

```text
1/4
1/6
1/8
1/12
1/16
1/24
1/32
1/48
```

从而在表达能力和序列长度之间取得相对合理的平衡。

如果原始 BPM 不在合适范围内，我们会通过乘除 2 的幂将其 canonicalize 到 `[120, 240)` 范围内，并记录对应的 `bpm_scale_exp`。这样做的目的不是改变音频播放速度，也不是拉伸谱面时间，而是让谱面在统一的 beat grid 密度下表示，便于模型学习节奏位置。

时间换算为：

```text
tick_ms = 60000 / canonical_bpm / 48
absolute_tick = round((time_ms - offset_ms) / tick_ms)
time_ms = offset_ms + absolute_tick * tick_ms
```

其中 `absolute_tick = 0` 对应 `offset_ms`，并不代表音乐开始或谱面开始。

> **【实现注记】** *(§2 Beat grid)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 48 tick/beat, 192 tick/bar | ✅ | `osu/grid_config.py:3-5` — `TICKS_PER_BEAT`, `TICKS_PER_BAR` |
> | BPM → `[120,240)` + scale_exp | ✅ | `osu/bpm.py:6-18` — `canonicalize_bpm()` |
> | tick_ms / ms↔tick 公式 | ✅ | `osu/row_tokens.py:64-70`（`beat_ms`, `tick_ms`）；`91-96`（`ms_to_tick`, `tick_to_ms`） |
> | offset 为 grid 原点 | ✅ | `ms_to_tick` 减 `offset_ms`；注释见 `osu/tick_range.py` |
> | 负 tick | ✅ | `osu/tick_range.py:19-23` — `split_absolute_tick()`；测试 `tests/test_row_tokens_v2.py` |
> | 从 .osu 读 timing | ✅ | `CanonicalTiming.from_timing_points()` — `row_tokens.py:72-84`；底层 `osu/timing.py`、`osu/parser.py`、`osu/schema.py` |
> | 元数据写入 cond_vec | ✅ | `data/chart_stats.py` → `difficulty/chart_meta.py` → `cond_vec.py:76-79` |
>
> **⚠️ 差异：** 已通过 filter 保证 constant BPM；`from_timing_points` 仍只读第一个 uninherited 点（多段 timing 谱已被 filter 掉，正常路径无影响）

---

## 3. 音频特征

音频侧输入与谱面使用同一套 canonical beat grid 对齐。我们将音频特征同步到 tick 级，使得 Encoder 的输入序列长度与当前窗口的 tick 数一致。

第一阶段使用传统、稳定且可解释的音频特征：

```text
128-dim log-Mel spectrogram
1-dim onset strength
1-dim RMS energy
12-dim chroma
```

合计：

```text
142 dim / tick
```

这些特征分别提供不同类型的信息：

```text
log-Mel:
  频谱内容、音色、鼓、人声、合成器、整体音乐纹理

onset strength:
  声音开始点、瞬态、鼓点、采音位置

RMS:
  局部能量、强弱、高潮/低潮、密度变化

chroma:
  音高类别、和声变化、旋律/和弦结构
```

模型的 Encoder 输入不是简单的音频特征本身，而是音频内容、时间位置和条件信息的组合：

```text
audio_embedding
+ audio_time_embedding
+ condition_embedding
```

其中 `audio_time_embedding` 至少包括：

```text
窗口内绝对 tick 位置
bar 内位置 pos_in_bar
beat 内位置 pos_in_beat
bar 内第几个 beat
```

这样模型不仅知道当前 tick 附近的声音内容，也知道它处在小节和节拍结构中的哪个位置。

> **【实现注记】** *(§3 音频特征)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 142-d = 128+1+1+12 | ✅ | `audio/tick_features.py:19-22` — `V2_FEATURE_DIM` |
> | 22.05 kHz | ✅ | `tick_features.py:19` — `V2_SAMPLE_RATE` |
> | tick 级对齐 | ✅ | `compute_tick_grid_features()` — `tick_features.py:33-88`；缓存 `data/audio_grid.py` |
> | 窗口切片 | ✅ | `data/audio_grid.py` — `slice_audio_window()`；训练样本 `data/chart_bundle.py:build_sample_from_bundle` |
> | audio_proj + time emb + cond | ✅ | `training/model.py:124-139` — `encode_audio()` |
> | 窗口内 tick 位置 | ✅ | `model.py:112-121` — `audio_abs_pos(t)`，`t = 0…T-1` |
> | pos_in_bar / pos_in_beat / beat_in_bar | ✅ | 同上，用 `TICKS_PER_BAR`、`TICKS_PER_BEAT` 分解 |
> | cond 注入 encoder | ✅ | `model.py:138-139` — `x += cond_h` |
>
> **⚠️ 与 OVERVIEW 未写明的实现细节（磁盘上 3567 张预计算 grid 均为此版本）：**
> - `AUDIO_FEATURE_SPEC_VERSION = 1`（`tick_features.py:25`）
> - hop ≈ `tick_ms/4`（动态），非固定 hop=512
> - frame→tick 用 `floor((frame_ms - offset) / tick_ms)`（`tick_features.py:75`）
> - mel / onset / rms / chroma **全部 mean 池化**（含 onset，非 max）
> - `audio_abs_pos` 是**窗口内** 0…T−1，不是整曲 global tick 索引
>
> **脚本：** `scripts/precompute_audio_grid.py`、`scripts/audit_audio_grid.py`  
> **音频加载：** `audio/loader.py` — 从 set 目录取 mono mp3（预计算用）  
> **测试：** `tests/test_v2_pipeline.py`（142 维、负 tick 切片）

---

## 4. 条件向量 cond_vec

`cond_vec` 用来控制生成谱面的整体目标。它不作为 token 放入 Decoder 序列，而是作为连续向量经过 MLP 映射为 `condition_embedding`，再注入 Encoder 和 Decoder。

条件向量可以包含：

```text
预期难度
长条比例
长条覆盖率
键型比例
键型难度
Etterna MSD 相关维度
BPM 相关信息
特征可用性 mask
```

例如：

```text
osu_sr_norm
hold_ratio
hold_coverage
pattern analyzer 的 Stream / Chordstream / Jacks / Coordination / Density / Wildcard 比例
Etterna overall / stream / jumpstream / handstream / stamina / jack / chordjack / technical
canonical_bpm_norm
bpm_scale_exp_norm
analyzer_available
msd_available
```

这些条件大多是全局目标，而不是局部 token。因此我们不把它们写进谱面 token 序列，而是将其编码为 `cond_emb`，并加到所有 Encoder audio tokens 和所有 Decoder chart tokens 上：

```text
encoder_x += cond_emb
decoder_x += cond_emb
```

这样每个音频位置和每个生成步骤都能够访问目标难度、长条比例和键型倾向。

> **【实现注记】** *(§4 cond_vec)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 23 维 float，非 token | ✅ | `data/cond_vec.py:9` — `COND_VEC_DIM = 23` |
> | 字段名与 OVERVIEW 示例 | ✅ 大体一致 | `cond_vec.py:12-36` — `COND_VEC_NAMES`（固定顺序，**勿随意重排**） |
> | MLP → cond_emb | ✅ | `training/model.py:70-75` — `cond_proj`: Linear→GELU→Linear→**LayerNorm** |
> | encoder 全 tick 加 cond | ✅ | `model.py:138-139` |
> | decoder 全 token 加 cond | ✅ | `model.py:170-171` |
> | 构建来源 | ✅ | `difficulty/chart_meta.py` — `compute_chart_meta()` → `build_cond_vec()` |
> | osu SR | ✅ | `difficulty/official_sr.py` |
> | hold 比例/覆盖 | ✅ | `difficulty/hold_ratio.py` |
> | Etterna MSD | ✅ | `difficulty/minacalc.py` |
> | pattern analyzer | ✅ | `analysis/pattern_features.py` → `pattern_analyser/*`（vendored，见 [附录 A.6](#a6-audio2mapanalysis--pattern_analyser--键型分析--cond_vec)） |
> | 离线批量 meta | — | `scripts/compute_chart_meta.py` |
>
> **索引顺序（实现固定，OVERVIEW 未写序号）：**
>
> | idx | 字段 |
> |-----|------|
> | 0 | osu_sr_norm（SR/10 clip 0–1.5） |
> | 1-2 | hold_ratio, hold_coverage |
> | 3-10 | analyzer_ln_percent, hb_row, stream, chordstream, jacks, coordination, density, wildcard |
> | 11-18 | etterna overall…technical（/40 clip） |
> | 19-20 | canonical_bpm_norm, bpm_scale_exp_norm |
> | 21-22 | **analyzer_available, msd_available**（flag 在末尾） |
>
> **⚠️ 差异：** 训练 preload 默认 `skip_msd=True`（`data/chart_bundle.py`），MSD 维常为 0 但 flag 仍写入；analyzer 不可用时 3–10 为 0、flag=0

---

## 5. 模型架构

鉴于谱面生成具有明显的自回归结构，并且与音乐转录、伴奏生成等任务在形式上相似，我们选择 Transformer Encoder-Decoder 作为基础架构。

整体结构为：

```text
audio tick features + time embeddings + cond embedding
        ↓
non-causal Transformer Encoder
        ↓
encoded audio memory
        ↓ cross-attention
causal Transformer Decoder
        ↓
chart token logits
```

Encoder 负责建模当前音乐窗口中的完整音频信息。由于这是离线谱面生成，模型在生成某个位置的谱面时可以看到整个音频窗口，因此 Encoder 使用 non-causal self-attention。

Decoder 负责自回归生成谱面 token。Decoder 的 self-attention 是 causal 的，只能看到已经生成的 chart token prefix；但它可以通过 cross-attention 访问 Encoder 输出的完整音频 memory。

这种结构与 prefix-LM 不同。我们不将：

```text
cond token + audio tokens + chart tokens
```

拼成一条序列做 causal language modeling，而是明确区分：

```text
audio encoder
chart decoder
cross-attention
```

这样更符合任务结构，也更利于模型在生成每个谱面事件时对齐音频上下文。

> **【实现注记】** *(§5 模型架构)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | **正式** enc_dec | ✅ | `training/model.py:39-284` — `AudioChartModel` |
> | non-causal audio encoder | ✅ | `nn.TransformerEncoder` — `model.py:77-85, 144` |
> | causal decoder + cross-attn | ✅ | `nn.TransformerDecoder` — `model.py:90-98, 177-183` |
> | cond 非 prefix token | ✅ | broadcast add，见 §4 |
> | `build_model(architecture="enc_dec")` | ✅ | `model.py:519-548` |
> | 正式训练强制 enc_dec | ✅ | `scripts/train_v2.py:120-121` |
>
> **⚠️ 代码中仍保留 legacy ablation（OVERVIEW 明确不要的主路径）：**
> - `PrefixLMAudioChartModel` — `model.py:296-516`：单条 causal 序列 `[cond; audio; chart]`
> - 支持 `audio_pooling="bar"`（enc_dec **始终 tick**）
> - 旧 checkpoint 无 `architecture` 字段时 `load_checkpoint` 默认 **prefix_lm**（`model.py:557-559`）
> - 过拟合/调试脚本 `--architecture prefix_lm` 仍可用
>
> **默认规模（CLI 可改）：** `train_v2.py` 默认 d=512, enc=4, dec=6, heads=8；`AudioChartModel.__init__` 代码默认 d=256, layers=4  
> **上限：** `MAX_AUDIO_TICKS = 16×192`（`model.py:28`），`MAX_DECODER_LEN = 1024`  
> **测试：** `tests/test_model_enc_dec.py`

---

## 6. 谱面 token 表示

为了让模型更容易学习 4K mania 中的键形规律，我们使用 row-level event token，而不是逐 lane 单独生成事件。

一个窗口内的谱面 token 序列为：

```text
<BOS>
<ROW_abcd>
<BAR>
<POS_x> <ROW_abcd>
<POS_x> <ROW_abcd>
...
<BAR>
<POS_x> <ROW_abcd>
...
<EOS>
(<PAD> ...)
```

其中：

```text
<BOS>
  序列开始 token

<ROW_abcd> after BOS
  窗口起点的 initial active hold state

<BAR>
  小节推进 token，即使该小节没有任何事件也要输出

<POS_x>
  当前小节内的位置，x ∈ {0, 1, ..., 191}

<ROW_abcd>
  当前 tick 上的 4K row event

<EOS>
  当前窗口结束

<PAD>
  batch padding token，只用于补齐 batch，不表示谱面内容
```

> **【实现注记】** *(§6 token 表示)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 序列形状 BOS→ROW→BAR→(POS ROW)*→EOS | ✅ | `osu/row_tokens.py:274-294` — `encode_window_tokens()` |
> | POS 0…191 | ✅ | `pos_token()` — `row_tokens.py:138-141`；`TICKS_PER_BAR=192` |
> | 词表 | ✅ | `build_vocab()` — `row_tokens.py:433-450` |
> | 特殊 token ID | ✅ | PAD=0, BOS=1, EOS=2, BAR=3（`row_tokens.py:434-438`） |
> | ROW 625 + POS 192 + 4 special = **821** | ✅ | 测试 `tests/test_row_tokens_v2.py:31-32` |
> | PAD 仅 collate | ✅ | 真值序列无 PAD；`training/collate.py` — `pad_batch()` 补尾 |
> | .osu ↔ tokens 往返 | ✅ | `beatmap_to_window_tokens` / 推理侧 `tokens_to_notes` — `round_trip.py` |

---

## 7. ROW token 语义

`<ROW_abcd>` 中的 `a,b,c,d` 分别对应 4 个 lane。每个 lane 的状态为：

```text
0 = empty
1 = tap
2 = hold_start
3 = hold_active
4 = hold_end
```

例如：

```text
<ROW_1001>
  L1 tap + L4 tap

<ROW_0200>
  L2 hold_start

<ROW_0301>
  L2 hold_active + L4 tap

<ROW_0400>
  L2 hold_end
```

使用 ROW token 的原因是，osu!mania 4K 的键形往往以同一时刻的 row 为基本单位，例如 chord、jump、hand、LN+tap、jack 等。将同一 tick 上 4 个 lane 的状态合并为一个 token，有助于模型直接学习 row-level pattern，而不是在多个 lane event 之间间接组合。

ROW 词表大小为：

```text
5^4 = 625
```

在可接受范围内，同时表达能力足够覆盖 tap、hold start、hold body active 和 hold end。

> **【实现注记】** *(§7 ROW 语义)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | LaneState 0–4 | ✅ | `row_tokens.py:46-52` — `LaneState` enum |
> | 5^4=625 ROW token | ✅ | `build_vocab()` 枚举 4 位 0–4 |
> | REAL_EVENT {1,2,4} | ✅ | `REAL_EVENT_STATES` — `row_tokens.py:42` |
> | 冲突检测（同 tick 双事件） | ✅ | `_merge_lane()` — `row_tokens.py:173-177` |
> | 诊断统计 | — | `scripts/analyze_row_token_stats.py` |
>
> **⚠️ 解码候选集：** `event_row_token_ids()` 排除「仅 initial 合法」的 ROW（约 609 个 event row vs 625 总数）— `row_tokens.py:458-468`

---

## 8. Initial ROW

序列开头的第二个 token：

```text
<ROW_abcd>
```

不是普通事件，而是窗口起点的初始 hold active 状态。

它只允许：

```text
a,b,c,d ∈ {0, 3}
```

也就是说，每个 lane 要么为空，要么已经处于 hold active 状态。

例如：

```text
<BOS>
<ROW_0000>
```

表示窗口开始时没有任何 lane 处于 hold 中。

```text
<BOS>
<ROW_0300>
```

表示窗口开始时 L2 已经有一个 hold 从窗口之前延续进来。

这个 initial ROW 是外部根据窗口起点计算出来的 prompt token，不是模型需要预测的目标。训练时 `<BOS>` 和 initial ROW 都不计入 loss。

> **【实现注记】** *(§8 Initial ROW)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 仅 {0,3} | ✅ | `INITIAL_LANE_STATES` — `row_tokens.py:43`；`is_initial_row()` — `153-155` |
> | 由窗口前 hold 状态计算 | ✅ | `initial_row_from_active()` — `161-164`；训练 `chart_bundle.py` 窗口起点 |
> | 推理 overlap 更新 initial | ✅ | `training/inference.py:383` — `initial_row_at_tick(all_notes, …)` |
> | loss 不计 BOS + initial ROW | ✅ | `data/window_sampler.py:25-33` — `build_loss_mask()` |
> | initial ROW 词表子集 | ✅ | `initial_row_token_ids()` — `row_tokens.py:471-478`（解码 prompt 用） |

---

## 9. 普通事件 ROW

普通事件以：

```text
<POS_x> <ROW_abcd>
```

的形式出现。

一个普通 ROW 至少要包含一个真实事件：

```text
tap
hold_start
hold_end
```

也就是说，普通事件 ROW 中至少有一个 lane 的状态属于：

```text
{1, 2, 4}
```

如果某个 tick 只有 hold_active，而没有 tap、hold_start 或 hold_end，则不输出这个 tick。

例如，L2 从 POS_48 hold 到 POS_144，且 L4 在 POS_96 有 tap：

```text
<POS_48>  <ROW_0200>
<POS_96>  <ROW_0301>
<POS_144> <ROW_0400>
```

其中 POS_96 的 L2 使用 state 3，是为了告诉模型此时 L2 仍处于 hold active 状态，而 L4 同时有 tap。

如果 POS_96 没有 L4 tap，则不会输出：

```text
<POS_96> <ROW_0300>
```

因为这只是 hold body，并不是新的谱面事件。

> **【实现注记】** *(§9 普通事件 ROW)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | is_event_row：含 {1,2,4} | ✅ | `row_tokens.py:157-158` |
> | 纯 hold body 不 emit | ✅ | `raw_events_to_bar_rows()` — `245-248`；`_build_event_row()` — `218-230` |
> | 共事件 tick 填 state 3 | ✅ | `_build_event_row()` 在 active hold lane 上写 `HOLD_ACTIVE` |
> | OVERVIEW 三段 hold 例子 | ✅ | 测试 `tests/test_row_tokens_v2.py:35-56` |
> | POS 严格递增（训练数据） | ✅ | tokenize 按 tick 排序产出；合法性 `validate_token_sequence()` — `370-430` |

---

## 10. BAR token 与空小节

`<BAR>` 表示进入一个新小节。即使该小节没有任何事件，也必须输出 `<BAR>`。

例如：

```text
<BAR>
<POS_0> <ROW_1000>
<BAR>
<BAR>
<BAR>
<POS_48> <ROW_0100>
```

这里中间两个 `<BAR>` 代表空小节。

这样设计的原因是，空小节也是时间结构的一部分。如果不输出空小节，Decoder 无法知道中间跳过了多少小节。空小节不使用 `<PAD>` 表示；`<PAD>` 只用于 batch 内补齐序列长度。

> **【实现注记】** *(§10 BAR / 空小节)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 每 bar 至少一个 `<BAR>` | ✅ | `encode_window_tokens()` — `row_tokens.py:274-294` 按 bar 循环 |
> | 空 bar 仅 `<BAR>` | ✅ | 无事件时只 append `TOKEN_BAR` — `286-288` |
> | 解码时空 bar 可 `<BAR>` 后直接下一 `<BAR>` 或 EOS | ✅ | `ChartDecodeState.allowed_token_ids()` — `decode.py:69-74, 87-90` |

---

## 11. PAD token

`<PAD>` 是训练时 batch padding 用的 token。由于不同窗口的 token 长度不同，训练时需要将它们补齐到同一长度：

```text
sample A:
<BOS> <ROW_0000> <BAR> <POS_0> <ROW_1000> <EOS> <PAD> <PAD>

sample B:
<BOS> <ROW_0000> <BAR> <POS_0> <ROW_1000> <POS_24> <ROW_0100> <EOS>
```

`<PAD>` 不表示任何谱面内容，也不参与 loss。推理时应禁止模型生成 `<PAD>`。

> **【实现注记】** *(§11 PAD)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | collate 补 PAD | ✅ | `training/collate.py` — `pad_batch()`；`attn_mask` 标记有效长度 |
> | PAD 不参与 loss | ✅ | collate 将 pad 位置 `loss_mask=0`；`model.py:compute_loss` 用 mask |
> | 推理禁止 PAD | ✅ | `ChartDecodeState.allowed_token_ids()` 从不含 PAD/BOS（`decode.py:64-91`） |
> | 真值 token 串无 PAD | ✅ | 仅 batch 维 padding |

---

## 12. 训练方式

训练时从一首谱面中采样一个连续窗口。窗口长度可以先固定为 8 bars 用于 debug，后续扩展为可变窗口长度，例如：

```text
8 bars
12 bars
16 bars
```

一个训练样本包含：

```text
audio window
chart token window
cond_vec
```

训练采用 teacher forcing。Decoder 输入为前缀 token，预测下一个 token。

对于 token 序列：

```text
<BOS>
<ROW_initial>
<BAR>
...
<EOS>
```

loss 规则为：

```text
<BOS>:
  不计入 loss

<ROW_initial>:
  不计入 loss

从第一个 <BAR> 开始:
  计入 loss

<EOS>:
  计入 loss

<PAD>:
  不计入 loss
```

也就是说，模型学习的是：

```text
给定音频窗口、条件向量、BOS 和初始 hold 状态，
自回归生成整个窗口内的谱面 token。
```

> **【实现注记】** *(§12 训练)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | 连续窗口采样 | ✅ | `data/window_sampler.py` — `sample_window_bars()`、`train_sample_bar_range()` |
> | 样本三元组 audio+tokens+cond | ✅ | `data/chart_bundle.py:166-199` — `build_sample_from_bundle()` |
> | Dataset / collate | ✅ | `training/dataset.py` — `Audio2MapV2Dataset`；`training/collate.py` — `pad_batch` |
> | 低层 sample 构建 | ✅ | `data/v2_dataset.py` — `build_training_sample`（`chart_bundle` 亦调用） |
> | 合格谱枚举 | ✅ | `data/eligible_charts.py` — `list_eligible_osu_paths()` |
> | 数据根路径 | ✅ | `utils/paths.py` — `AUDIO2MAP_DATA_ROOT`、checkpoint 默认目录 |
> | teacher forcing | ✅ | `model.py:147-184` — 输入 `token_ids[:,:-1]` 预测 next |
> | loss mask 规则 | ✅ | `window_sampler.py:25-33`；测试 `tests/test_v2_pipeline.py:73-82` |
> | 可变 window 8/12/16 | ⚠️ | **基础设施支持** `window_bars_choices`（`dataset.py` DatasetConfig）；`train_v2.py` 默认 `"8,12,16"`；但 `grid_config.py:8` 库默认仍 `(8,)` |
> | 正式训练脚本 | ✅ | `scripts/train_v2.py` — 多 chart、enc_dec、bf16、preload |
> | 单谱过拟合 | — | `scripts/train_debug_overfit.py` |
> | eager preload vs lazy | ⚠️ | `--lazy` 极慢；全量 11927 谱 preload 约 1h+ 后才开始 GPU 步 |
>
> **margin：** 窗口外扩 `pre_event_margin_bars=4`, `post_event_margin_bars=4`（`window_sampler.py:17-18`）— OVERVIEW 未写

---

## 13. 推理方式

推理时，首先测定或输入音乐的 BPM 和 offset，构建 beat grid，并提取 tick-level 音频特征。

然后模型使用自回归方式生成谱面 token：

```text
prompt:
<BOS>
<ROW_initial>

generate:
<BAR>
<POS_x> <ROW_abcd>
...
<EOS>
```

对于整首歌生成，我们使用 overlap generation。也就是说，将整首音乐分成多个重叠窗口，每个窗口单独生成，然后只保留其中一段稳定区域进行拼接。

例如：

```text
window = context + keep + future
```

其中：

```text
context:
  已经生成的前文谱面，作为 prompt

keep:
  当前窗口真正保留到最终谱面的部分

future:
  未来音频 lookahead，用来减少窗口边界效应，但生成结果可丢弃
```

第一窗口没有历史 context，因此不应丢弃开头小节。后续窗口可以使用前一段已生成谱面作为 context。

生成时还需要 constrained decoding，以保证 token 序列合法。

> **【实现注记】** *(§13 推理)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | BPM/offset → grid + 特征 | ✅ | `scripts/infer_v2.py`；`data/audio_grid.py` |
> | prompt `<BOS><ROW_initial>` | ✅ | `decode.py:126-128` — `bos_initial_prefix()`；`inference.py:229-230` |
> | 自回归 `generate_window_tokens()` | ✅ | `training/inference.py:200-260` |
> | overlap 窗口划分 | ✅ | `OverlapConfig` — `inference.py:37-49`；`overlap_inference_bar_windows()` — `134-157` |
> | 首窗 keep 从 win_start | ✅ | 测试 `tests/test_decode_inference.py:59-60` |
> | 按 keep 区间拼接 notes | ✅ | `generate_chart_notes()` — `inference.py:376-405` 过滤 tick 范围 |
> | constrained decoding | ✅ | 见 §14 |
> | CLI | ✅ | `scripts/infer_v2.py`（默认 `audio_full` 至 mp3 末） |
>
> **❌ / ⚠️ 与 OVERVIEW 的重要差异：**
> - **Context 未作为 decoder token prefix：** `generate_window_tokens()` 支持 `prompt_token_ids`（`inference.py:218-228`），但 `generate_chart_notes()` **从未传入**前文 token；overlap 只靠 (1) 更新 `initial_row` (2) 按 keep 区间**裁 note** 拼接，**不是** teacher-forcing 式 context 续写
> - **future_bars：** `OverlapConfig` 默认 `future_bars=0`（`inference.py:41`）；`infer_v2.py` 用 `window−context−keep` 计算，默认 8−4−4=0；音频窗口仍覆盖整窗，但「future 段 token 丢弃」无独立实现
> - 解码未自然结束时会**强制 append EOS**（`inference.py:257-258`）

---

## 14. Constrained decoding

在解码过程中维护：

```text
当前小节内 POS
当前 active_hold 状态
当前语法状态
```

约束包括：

```text
1. <PAD> 不允许生成
2. <BOS> 只出现在序列开头
3. <POS_x> 在同一 BAR 内必须严格递增
4. <POS_x> 后必须接一个 event ROW
5. event ROW 至少包含一个 {tap, hold_start, hold_end}
6. tap 不能出现在已经 active hold 的 lane 上
7. hold_start 不能出现在已经 active hold 的 lane 上
8. hold_end 必须出现在 active hold 的 lane 上
9. hold_active 只能出现在 active hold 的 lane 上
```

ROW 合法性逐 lane 检查：

```text
0 empty:
  always legal

1 tap:
  active_hold[lane] must be False

2 hold_start:
  active_hold[lane] must be False

3 hold_active:
  active_hold[lane] must be True

4 hold_end:
  active_hold[lane] must be True
```

生成 ROW 后更新 active hold 状态：

```text
hold_start -> active_hold = True
hold_end   -> active_hold = False
tap        -> unchanged
empty      -> unchanged
active     -> unchanged
```

> **【实现注记】** *(§14 Constrained decoding)*
>
> | # | 约束 | 状态 | 代码位置 |
> |---|---|---|---|
> | 1 | 禁止 PAD | ✅ | `decode.py:64-91` — allowed set 不含 PAD |
> | 2 | BOS 仅 prefix | ✅ | 解码从 `bos_initial_prefix` 之后开始，状态机不接受 BOS |
> | 3 | POS 严格递增 | ✅ | `decode.py:83-85` |
> | 4 | POS 后必 ROW | ✅ | `expect_row` 标志 — `decode.py:76-80, 108-109` |
> | 5 | event ROW 含 {1,2,4} | ✅ | 候选来自 `event_row_token_ids` + `is_event_row` |
> | 6–9 | lane 合法性 | ✅ | `is_legal_row()` — `row_tokens.py:341-359`；解码过滤 `decode.py:77-80` |
> | — | hold 状态更新 | ✅ | `apply_row()` — `row_tokens.py:361-368`；`observe()` — `decode.py:111-114` |
> | — | 空小节收束 | ✅ | `observe_and_advance_bar_if_needed()` — `decode.py:118-123` |
> | — | 采样/top-p 仅在 allowed 集 | ✅ | `inference.py:160-191` |
>
> **⚠️** `validate_token_sequence()`（`row_tokens.py:370-430`）实现了完整序列校验，但**导出/推理管线默认不调用**

---

## 15. 导出谱面

模型生成 token 序列后，需要将其转换回 osu!mania hit objects。

其中：

```text
state 1 tap:
  生成普通 note

state 2 hold_start:
  记录 hold 起点

state 4 hold_end:
  与之前的 hold_start 配对，生成 LN

state 3 hold_active:
  不直接生成 hit object，只表示该 tick 上有 hold 从之前延续
```

导出前需要做合法性检查和后处理：

```text
检查 hold_start / hold_end 是否配对
删除或修复未闭合 hold
删除同 lane overlap
删除 duration <= 0 的 hold
过滤音频范围外事件
转换 tick 回 time_ms
写入 .osu
```

> **【实现注记】** *(§15 导出)*
>
> | 项 | 状态 | 代码位置 |
> |---|---|---|
> | state→note 语义 | ✅ | `osu/round_trip.py:54-77` — `tokens_to_notes()` |
> | tick→time_ms | ✅ | `row_tokens.tick_to_ms()` |
> | 写 .osu | ✅ | `osu/export.py:51-95` — `export_beatmap_notes()` 克隆模板替换 `[HitObjects]` |
> | y=192, sample 0:0:0:0: | ✅ | `export.py:10-11, 30-35` |
> | BeatmapID=0 等编辑器兼容 | ✅ | `_sanitize_export_line()` — `export.py:38-48` |
> | 负时间过滤 | ✅ | `filter_notes_for_export(min_time_ms=0)` — `export.py:14-27` |
> | duration≤0 hold 过滤 | ✅ | 同上 `end <= time` 丢弃 |
>
> **❌ OVERVIEW §15 后处理大多未单独实现：**
> - hold 配对失败：**孤儿 hold_end 静默丢弃**（`round_trip.py:77` 注释），无修复
> - 同 lane overlap 删除 — **无**
> - 音频范围外事件过滤 — **仅** `min_time_ms≥0`，无 mp3 上界
> - 导出前 `validate_token_sequence` — **未调用**
> - 批量对比包 — `scripts/batch_pack_osz.py`

---

## 16. 评估指标

评估指标后续仍需进一步讨论，但至少应包含以下几类。

### 16.1 合法性

```text
非法 ROW 数量
未闭合 hold 数量
同 lane overlap 数量
POS 顺序错误数量
导出 .osu 是否可被解析
```

### 16.2 条件符合度

```text
目标难度 vs 生成谱面难度
目标 hold_ratio vs 生成 hold_ratio
目标 hold_coverage vs 生成 hold_coverage
目标 pattern distribution vs 生成 pattern distribution
目标 MSD vs 生成 MSD
```

### 16.3 音乐对齐

```text
note tick 附近 onset strength
note density 与 RMS / energy 的相关性
强拍 / 弱拍 / 小节位置分布
POS usage histogram
```

### 16.4 结构与多样性

```text
bar-level note density
重复 1-bar / 2-bar pattern 比例
n-gram repetition rate
autocorrelation
段落间密度变化
```

### 16.5 与参考谱面对比

在 validation/test 中，如果存在参考谱面，可以比较：

```text
token-level accuracy
BAR / POS / ROW accuracy
hold_start recall
hold_end recall
ROW lane-state accuracy
density curve similarity
pattern distribution distance
```

但需要注意，谱面生成不是唯一答案任务，因此不能只用 token accuracy 判断质量。更重要的是合法性、条件符合度、音乐对齐和结构合理性。

> **【实现注记】** *(§16 评估)*
>
> | 小节 | 状态 | 已有代码 |
> |---|---|---|
> | **16.1 合法性** | ❌ 无汇总指标 | 库函数 `validate_token_sequence()` 存在；**无** eval 脚本聚合非法 ROW/hold/overlap |
> | **16.2 条件符合度** | ❌ | 未实现 target vs gen 的 SR/hold/pattern/MSD 对比 |
> | **16.3 音乐对齐** | ❌ | 未实现 onset/RMS/强拍分布类指标 |
> | **16.4 结构/多样性** | ⚠️ 部分 | `eval/degeneracy.py` — POS 直方图、bar 密度自相关、ROW bigram 重复、collapse 启发 |
> | **16.5 参考对比** | ⚠️ 部分 | `eval/token_accuracy.py`；`eval/note_match.py` — tick note P/R/F1；`eval/chart_eval.py` — roundtrip、teacher forcing、generate |
> | **16.5 未做** | ❌ | density curve similarity、pattern distribution distance |
>
> **CLI：** `scripts/eval_v2.py`（roundtrip / window / teacher / infer / generate）  
> **测试：** `tests/test_eval.py`、`tests/test_difficulty.py`（SR/MSD/hold）  
> **infer 附带：** `infer_v2.py` 可报 `degeneracy` 统计，非完整 §16

---

## 17. 总结

整体方案可以概括为：

```text
给定 constant BPM + meter=4 的 osu!mania 4K 音乐，
先通过 BPM 和 offset 建立 1/48 beat grid，
再将音频特征对齐到 tick 级，
使用 Transformer Encoder-Decoder 建模：

P(chart | audio, cond_vec)

Encoder 读取完整音频窗口；
Decoder 自回归生成 row-level chart tokens；
condition vector 控制难度、长条比例和键型倾向；
constrained decoding 保证生成谱面合法；
overlap generation 用于整首歌长序列生成。
```

这种设计牺牲了一部分通用性，但它更贴近 osu!mania 4K 的谱面结构，也更利于模型学习 beat-based 节奏、row-level 键形和 hold 状态变化。

> **【实现注记】** *(§17 总结 — 实现完成度快照)*
>
> **与 OVERVIEW 核心设计一致的部分：** Phase-1 过滤、1/48 beat grid、142-d tick 音频、enc_dec + cond 注入、821-token ROW 词表、initial/event ROW 规则、空 BAR、teacher forcing + loss mask、约束解码 FSM、overlap 窗口调度、基础 .osu 导出。
>
> **审阅时建议优先核对的缺口：**
> 1. overlap **context token prompt** 未接入（仅 note 裁剪拼接）
> 2. 导出缺少完整后处理（§15）
> 3. 评估缺少合法性/条件/对齐（§16.1–16.3）
> 4. 磁盘 audio grid 为 feature **v1**（mean 池化、动态 hop），OVERVIEW 未限定细节
> 5. `prefix_lm` legacy 仍在仓库中
>
> **代码导航速查（主路径）：** 完整文件列表见 **[附录 A](#附录-a仓库代码全量索引)**。
>
> | 模块 | 路径 |
> |------|------|
> | 过滤 | `audio2map/data/chart_filter.py` |
> | Beat grid / token | `audio2map/osu/row_tokens.py` |
> | 音频特征 | `audio2map/audio/tick_features.py` |
> | cond_vec | `audio2map/data/cond_vec.py` |
> | 训练样本 | `audio2map/data/chart_bundle.py` |
> | 模型 | `audio2map/training/model.py` — `AudioChartModel` |
> | 解码 FSM | `audio2map/training/decode.py` |
> | 推理/overlap | `audio2map/training/inference.py` |
> | 导出 | `audio2map/osu/round_trip.py`, `export.py` |
> | 脚本 | `scripts/train_v2.py`, `infer_v2.py`, `eval_v2.py` |

---

## 附录 A：仓库代码全量索引

> Agent 2026-06-01 盘点 `/root/audio2map`。**凡下表列出的路径均在注记体系中至少提及一次。**  
> **状态：** `v2` = 当前 v2 主路径会 import/调用；`legacy` = v1 遗留，不参与 v2 训练；`tool` = CLI/离线工具；`test` = pytest；`cfg` = 参考配置（脚本不自动加载）；`vendored` =  vendored 子包。

### A.1 包根

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `audio2map/__init__.py` | 包标识 | — | v2 |

### A.2 `audio2map/osu/` — 解析、节拍网格、token、导出

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `grid_config.py` | `TICKS_PER_BEAT/BAR`、默认 window bars | §2, §12 | v2 |
| `bpm.py` | `canonicalize_bpm()` → `[120,240)` | §2 | v2 |
| `row_tokens.py` | `CanonicalTiming`、ROW/POS 词表、编解码、合法性 | §2, §6–11, §14 | v2 |
| `tick_range.py` | 负 tick、`audio_bar_range_ms`、训练/推理区间 | §2, §12, §13 | v2 |
| `timing.py` | `TimingSummary`、constant BPM/meter 判定 | §1, §2 | v2 |
| `schema.py` | `Beatmap`、`ManiaNote`、`TimingPoint` 数据结构 | §1, §15 | v2 |
| `parser.py` | `.osu` 全文解析 | §1, §15 | v2 |
| `mania.py` | 4K lane x 坐标、`is_mania_4k_sections` | §1, §15 | v2 |
| `round_trip.py` | `tokens_to_notes`、`quantize_notes` | §15 | v2 |
| `export.py` | `export_beatmap_notes`、编辑器兼容 sanitize | §15 | v2 |
| `__init__.py` | 子包导出 | — | v2 |

### A.3 `audio2map/audio/` — 音频 I/O 与特征

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `tick_features.py` | 142-d tick 特征、`AUDIO_FEATURE_SPEC_VERSION` | §3 | v2 |
| `loader.py` | `load_mono_audio`、找 set 内 `.mp3` | §3（数据准备） | v2 |

### A.4 `audio2map/data/` — 数据集、缓存、采集

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `chart_filter.py` | Phase-1  eligibility 过滤 | §1 | v2 |
| `eligible_charts.py` | 枚举合格 `.osu` 路径列表 | §1, §12 | v2 |
| `chart_stats.py` | 单谱 timing/统计 → meta 管线 | §2, §4 | v2 |
| `cond_vec.py` | 23-d `build_cond_vec` | §4 | v2 |
| `audio_grid.py` | grid 预计算/缓存/切片 `slice_audio_window` | §3, §12, §13 | v2 |
| `window_sampler.py` | 窗口采样、`build_loss_mask` | §12 | v2 |
| `chart_bundle.py` | preload bundle、`build_sample_from_bundle` | §12 | v2 |
| `v2_dataset.py` | 低层 `build_training_sample`（bundle 亦用） | §12 | v2 |
| `collector/sayobot.py` | Sayobot API 批量下载谱面 | §1（数据） | tool |
| `collector/cli.py` | 采集器 CLI 入口 | §1（数据） | tool |
| `collector/state.py` | 采集进度/状态持久化 | §1（数据） | tool |

### A.5 `audio2map/difficulty/` — 难度与 cond 字段来源

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `chart_meta.py` | `ChartMeta`、`compute_chart_meta` 聚合 | §4 | v2 |
| `official_sr.py` | osu 官方 Star Rating | §4, §16.2 | v2 |
| `hold_ratio.py` | LN 比例、`hold_coverage` 相关 | §4, §16.2 | v2 |
| `minacalc.py` | Etterna MSD 计算 | §4, §16.2 | v2 |
| `__init__.py` | 子包导出 | §4 | v2 |

### A.6 `audio2map/analysis/` + `pattern_analyser/` — 键型分析 → cond_vec

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `analysis/pattern_features.py` | 薄封装 `analyze_chart_patterns` | §4 | v2 |
| `pattern_analyser/__init__.py` | 导出 `analyze_osu`、`PatternFeatures` | §4 | vendored |
| `pattern_analyser/analyze.py` | 分析主入口 | §4 | vendored |
| `pattern_analyser/chart.py` | 内部 chart 表示 | §4 | vendored |
| `pattern_analyser/osu_parser.py` | analyser 专用 parser | §4 | vendored |
| `pattern_analyser/find_patterns.py` | 模式搜索 | §4 | vendored |
| `pattern_analyser/categorise.py` | Stream/Jacks/… 分类 | §4 | vendored |
| `pattern_analyser/patterns_def.py` | 模式定义 | §4 | vendored |
| `pattern_analyser/primitives.py` | 基元特征 | §4 | vendored |
| `pattern_analyser/clustering.py` | 聚类 | §4 | vendored |
| `pattern_analyser/summary.py` | 汇总统计 | §4 | vendored |
| `pattern_analyser/config.py` | analyser 配置 | §4 | vendored |
| `pattern_analyser/service.py` | 服务层封装 | §4 | vendored |
| `pattern_analyser/output_writer.py` | 结果写出 | §4 | vendored |
| `pattern_analyser/time_types.py` | 时间类型 | §4 | vendored |
| `pattern_analyser/card.py` | 卡片/展示辅助 | §4 | vendored |

### A.7 `audio2map/training/` — 模型、数据加载、解码、推理

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `model.py` | `AudioChartModel`、`PrefixLMAudioChartModel`、`build_model` | §3–5, §12 | v2 |
| `dataset.py` | `Audio2MapV2Dataset`、`DatasetConfig` | §12 | v2 |
| `collate.py` | `pad_batch`、PAD mask | §11, §12 | v2 |
| `decode.py` | `ChartDecodeState` 约束解码 FSM | §13, §14 | v2 |
| `inference.py` | overlap、`generate_window_tokens/chart_notes` | §13 | v2 |
| `__init__.py` | 训练 API 再导出 | §5 | v2 |

### A.8 `audio2map/eval/` — 评估（§16，大部分未实现）

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `chart_eval.py` | roundtrip / teacher / infer / generate 评估入口 | §16.5 | v2 |
| `token_accuracy.py` | BAR/POS/ROW/hold recall | §16.5 | v2 |
| `note_match.py` | tick 级 note P/R/F1 | §16.5 | v2 |
| `degeneracy.py` | POS 直方图、重复、collapse 启发 | §16.4 | v2 |
| `__init__.py` | 子包导出 | §16 | v2 |

### A.9 `audio2map/legacy/` — v1 遗留（10 ms 稀疏事件）

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `legacy/README.md` | 说明 v1 已归档 | §1 | legacy |
| `legacy/events.py` | 10 ms hop 稀疏 `EventType` | §1 | legacy |
| `legacy/tokens.py` | v1 token 词表 | §1 | legacy |
| `legacy/frames.py` | v1 音频帧特征 | §1 | legacy |
| `legacy/__init__.py` | 子包标识 | §1 | legacy |

### A.10 `audio2map/utils/`

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `paths.py` | `AUDIO2MAP_DATA_ROOT`、raw/processed/checkpoint 路径 | §12–13（数据根） | v2 |

### A.11 `scripts/` — CLI 入口

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `collect.py` | 启动 Sayobot 采集 | §1 | tool |
| `analyze_dataset.py` | 数据集 eligibility 统计 | §1 | tool |
| `analyze_timing.py` | raw 集 BPM/meter 分布分析 | §2 | tool |
| `compute_chart_meta.py` | 批量预计算 `ChartMeta` JSON | §4 | tool |
| `precompute_audio_grid.py` | 批量 tick audio grid | §3 | tool |
| `audit_audio_grid.py` | 校验 grid 缓存完整性 | §3 | tool |
| `train_v2.py` | 正式多 chart 训练 | §5, §12 | v2 |
| `train_debug_overfit.py` | 单谱过拟合调试 | §12 | v2 |
| `infer_v2.py` | 整曲推理 + 导出 `.osu`/`.osz` | §13, §15 | v2 |
| `eval_v2.py` | 评估 CLI | §16 | v2 |
| `batch_pack_osz.py` | 批量打 compare `.osz` 包 | §15 | tool |
| `analyze_row_token_stats.py` | ROW 词频/分布诊断 | §7 | tool |

### A.12 `tests/` — pytest

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `test_row_tokens_v2.py` | token 编解码、filter、负 tick | §1–11 | test |
| `test_osu_parser.py` | parser / 4K 校验 | §1, §15 | test |
| `test_v2_pipeline.py` | cond_vec、loss mask、audio 维、grid | §3–4, §12 | test |
| `test_model_enc_dec.py` | enc_dec 形状、checkpoint | §5 | test |
| `test_decode_inference.py` | 约束解码、overlap 首窗 | §13–14 | test |
| `test_eval.py` | eval 模块 smoke | §16 | test |
| `test_difficulty.py` | SR/MSD/hold_ratio/meta | §4, §16.2 | test |

### A.13 `configs/` — 参考 YAML（**脚本不自动读取**）

| 文件 | 作用 | OVERVIEW | 状态 |
|------|------|----------|------|
| `configs/README.md` | 说明 YAML 仅为参考 | §12 | cfg |
| `configs/data/v2.yaml` | window bars、margin、采样率 | §12 | cfg |
| `configs/data/collector.yaml` | 采集 delay 等 | §1 | cfg |
| `configs/train/train_multi.yaml` | 正式训练超参参考 | §12 | cfg |
| `configs/train/debug_overfit.yaml` | 过拟合超参参考 | §12 | cfg |

### A.14 仓库根（非 Python，审阅时可忽略实现）

| 文件 | 作用 |
|------|------|
| `README.md` | 仓库说明、命令速查 |
| `environment.yml` | conda 环境依赖 |
| `docs/*`（除本文件） | Agent 笔记 / 对话日志，**未确认** |

### A.15 按 OVERVIEW 章节反向查代码

| OVERVIEW | 涉及文件（完整路径见上表） |
|----------|---------------------------|
| §0 总览 | `train_v2.py`, `infer_v2.py`, `chart_bundle.py` → `model.py` → `inference.py` → `round_trip.py` → `export.py` |
| §1 任务限定 | `chart_filter.py`, `eligible_charts.py`, `mania.py`, `timing.py`, `parser.py`, `collect.py`, `collector/*`, `legacy/*` |
| §2 Beat grid | `grid_config.py`, `bpm.py`, `row_tokens.py`, `tick_range.py`, `chart_stats.py`, `analyze_timing.py` |
| §3 音频 | `tick_features.py`, `loader.py`, `audio_grid.py`, `precompute_audio_grid.py`, `audit_audio_grid.py` |
| §4 cond_vec | `cond_vec.py`, `chart_meta.py`, `official_sr.py`, `hold_ratio.py`, `minacalc.py`, `pattern_features.py`, `pattern_analyser/*`, `compute_chart_meta.py` |
| §5 架构 | `model.py`, `training/__init__.py`, `train_v2.py`, `train_debug_overfit.py` |
| §6–11 Token | `row_tokens.py`, `collate.py`, `decode.py`, `chart_tokens.md`（未确认速查） |
| §12 训练 | `window_sampler.py`, `v2_dataset.py`, `chart_bundle.py`, `dataset.py`, `collate.py`, `train_v2.py`, `train_debug_overfit.py`, `configs/train/*` |
| §13 推理 | `inference.py`, `decode.py`, `infer_v2.py`, `audio_grid.py` |
| §14 解码约束 | `decode.py`, `row_tokens.py`（`is_legal_row`, `validate_token_sequence`） |
| §15 导出 | `round_trip.py`, `export.py`, `infer_v2.py`, `batch_pack_osz.py` |
| §16 评估 | `eval/*`, `eval_v2.py` |
| 路径/数据根 | `utils/paths.py` |

---

> **其他文档：** [V2_MASTER_SPEC.md](V2_MASTER_SPEC.md)、[v2_spec.md](v2_spec.md) 等为 Agent 笔记/实现对照，**未确认**；仅作参考，不可当作权威规格。  
> **本页实现注记 + 附录 A** 同样为 Agent 对照产物，审阅请以正文 OVERVIEW 为准，注记供跳转代码与发现差异之用。
