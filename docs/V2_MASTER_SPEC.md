# Audio2Map v2 主规格：osu!mania 4K 谱面生成

> **⚠️ 未经 sjx 确认 — 不可当作权威规格。**  
> Agent 在会话中整理；可能与 [OVERVIEW.md](OVERVIEW.md) 或实际代码不一致。  
> **唯一可信方案文档：[OVERVIEW.md](OVERVIEW.md)。** 冲突以 OVERVIEW 为准；不确定先问 sjx。

最后同步：2026-06-01（Agent 维护，不保证正确）

---

## 0. 一句话目标

我们要做的是：

```text
P(chart | audio, BPM, offset, condition_vector)
```

具体任务：

```text
输入：
- mono 音频，22.05 kHz
- constant BPM
- offset_ms
- 全局条件向量 cond_vec
- osu!mania 4K

输出：
- osu!mania 4K chart
- 包含 tap 和 LN/hold
```

Phase 1 的限制：

```text
只做 osu!mania 4K
只做 constant BPM
只做 constant meter
只做 meter = 4
BPM/offset 由外部测定或从 osu timing point 读取
不做变 BPM
不做非 4/4
不生成 timing
不生成 hitsound
不生成 SV
不生成 kiai
不生成 metadata
```

---

# 1. 当前实现最大问题：不要把 prefix-LM 当主模型

正式主架构必须是：

```text
Audio Encoder:
  tick-level audio
  non-causal TransformerEncoder

Chart Decoder:
  autoregressive TransformerDecoder

Connection:
  decoder cross-attention to encoded audio memory

Condition:
  cond_emb 注入所有 encoder audio tokens 和所有 decoder chart tokens
```

也就是：

```text
audio_grid
  -> audio embedding + time embedding + cond embedding
  -> non-causal encoder
  -> encoded_audio_memory

chart prefix tokens
  -> token embedding + decoder position embedding + cond embedding
  -> causal decoder
  -> cross-attention(encoded_audio_memory)
  -> next chart token logits
```

**不要再把 `[cond ; audio ; chart]` 拼成一条序列做 causal LM。**

如果保留 prefix-LM，只能作为 **ablation baseline**，不能作为 v2 主线。

---

# 2–31. 完整条款

以下章节与 sjx 2026-06-01 同步的完整主规格一致。  
实现状态跟踪见 [V2_COMPLIANCE_AUDIT.md](V2_COMPLIANCE_AUDIT.md)。

| § | 主题 |
|---|------|
| 2 | 数据筛选 |
| 3 | BPM canonicalization |
| 4 | offset 语义（absolute_tick 可为负） |
| 5 | absolute bar / POS |
| 6 | Audio tick-grid 范围 |
| 7 | Audio feature（142 维，tick-centered pooling） |
| 8 | **严禁 bar-level audio pooling** |
| 9 | Tokenization（821 vocab，5-state ROW） |
| 10 | Empty bar 必须 emit `<BAR>` |
| 11 | Round-trip 要求 |
| 12 | Window sampling（无 empty-window rejection） |
| 13 | hold_coverage |
| 14 | Condition vector（23 维，cond_emb 注入 encoder/decoder 所有 token） |
| 15 | Pattern analyzer |
| 16 | Etterna MSD |
| 17 | Difficulty（osu_sr 为主） |
| 18 | **主模型 encoder-decoder 架构** |
| 19 | POS 为 token |
| 20 | 训练目标（continuous-window AR） |
| 21 | Window length（variable `window_bars_choices`） |
| 22 | max_token_len 统计 |
| 23 | 推理（overlap、first window、audio_full、采样） |
| 24 | Constrained decoding |
| 25 | Export / postprocess |
| 26 | 指标与诊断 |
| 27 | Sanity check 门禁 |
| 28 | 实施顺序 |
| 29 | 必须立刻纠正的点 |
| 30 | 与 Mapperatorinator 的关系 |
| 31 | **十条不可违背原则** |

---

## 31. 最终不可违背原则（检查清单）

```text
1. offset 是 beat-grid origin，不是音乐起点；tick 可以为负。
2. audio 必须 tick-level 输入；不能 bar-pool。
3. tokenization 是 BOS + initial ROW + BAR/POS/ROW + EOS。
4. initial ROW 是 prompt，不是预测目标。
5. condition 是 cond_vec -> cond_emb，注入 encoder/decoder 所有 tokens。
6. 主模型是真 encoder-decoder，不是 prefix-LM。
7. audio encoder 非因果，decoder 因果。
8. decoder cross-attend encoded audio memory。
9. 训练是 continuous-window AR，整 window chart tokens 都算 loss。
10. 推理 overlap 是 decoding strategy，不是训练目标。
```

**违反任一条 → 不是 v2 主方案。**

---

## 附录 A：§14.1 cond_vec 维度顺序（固定）

```text
0:  osu_sr_norm
1:  hold_ratio
2:  hold_coverage
3:  analyzer_ln_percent
4:  analyzer_hb_row_ratio
5:  analyzer_stream_ratio
6:  analyzer_chordstream_ratio
7:  analyzer_jacks_ratio
8:  analyzer_coordination_ratio
9:  analyzer_density_ratio
10: analyzer_wildcard_ratio
11: etterna_overall_norm
12: etterna_stream_norm
13: etterna_jumpstream_norm
14: etterna_handstream_norm
15: etterna_stamina_norm
16: etterna_jack_norm
17: etterna_chordjack_norm
18: etterna_technical_norm
19: canonical_bpm_norm
20: bpm_scale_exp_norm
21: analyzer_available
22: msd_available
```

## 附录 B：§18.5 推荐模型大小

debug：

```yaml
d_model: 256
nhead: 4
encoder_layers: 2
decoder_layers: 2
dim_feedforward: 1024
dropout: 0.1
```

正式起点：

```yaml
d_model: 256
nhead: 8
encoder_layers: 4
decoder_layers: 6
dim_feedforward: 1024
dropout: 0.1
```

## 附录 C：§7 Audio feature 参数

```yaml
sample_rate: 22050
n_fft: 2048
hop_length: 512
n_mels: 128
```

tick-centered pooling：`[tick_time - 0.5 tick, tick_time + 0.5 tick]`

- mel: mean
- onset: **max**
- rms: mean
- chroma: mean

缓存建议：`float16 [num_ticks, 142]`

**当前部署（2026-06-01）：** 磁盘 3567 grids 为 **feature spec v1**（动态 hop + floor 分桶）；代码 `AUDIO_FEATURE_SPEC_VERSION=1` 与之对齐。§7 完整参数为 **v2 目标**，切换前做小样本对比，不今晚全量 precompute。

---

完整 prose 版（§2–§30 逐节原文）见 git 历史或 sjx 2026-06-01 同步消息；  
**实现与合规以本文 + [V2_COMPLIANCE_AUDIT.md](V2_COMPLIANCE_AUDIT.md) 为准。**
