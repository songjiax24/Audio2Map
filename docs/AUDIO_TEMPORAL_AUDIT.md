# Audio 时间维压缩审计（2026-06-01）

> **结论：** 唯一致命的 **tick→bar** 压缩在 `AudioChartModel._pool_audio_by_bar`。  
> 预计算阶段的 frame→tick pooling 是保留 tick 分辨率的，不是 bar downsampling。

## 搜索范围

```text
mean( / avg_pool / adaptive_avg / reshape(..., 192 / TICKS_PER_BAR
_pool_audio / bar.pool / audio_stride
```

## 结果

| 位置 | 操作 | 输入粒度 | 输出粒度 | 是否问题 |
|------|------|----------|----------|----------|
| `audio2map/training/model.py` `_pool_audio_by_bar` | `view(..., window_bars, 192, F).mean(dim=2)` | **tick** (window_bars×192) | **bar** (window_bars) | **是 — 已改为 tick-level（`audio_pooling=tick`）** |
| `audio2map/audio/tick_features.py` | librosa 帧累加进 tick bin 后除以 count | STFT 帧 (~4 帧/tick) | **tick** | 否 — 目标就是 tick grid |
| `audio2map/data/audio_grid.py` `slice_audio_window` | 无压缩 | tick | tick | 否 |
| `audio2map/data/v2_dataset.py` | 无压缩 | tick | tick | 否 |
| `audio2map/training/inference.py` | 无压缩（传入 model 前） | tick | tick | 否 |

## 不存在

- `avg_pool` / `adaptive_avg_pool` on audio
- tick→beat stride downsampling
- 其他 `reshape(..., 192, ...).mean` 路径

## 模型 audio prefix 长度（修复后）

| `audio_pooling` | encoder prefix 长度 | 与 tick 数关系 |
|-----------------|----------------------|----------------|
| `bar`（旧 checkpoint） | `1 + window_bars` | **≠** tick count |
| `tick`（新默认） | `1 + window_ticks` | **==** tick count |

8-bar window：`window_ticks = 8 × 192 = 1536`，prefix = **1537**（含 cond）。

## 旧 checkpoint 兼容性

`step_20000.pt` 等旧权重在 `audio_pooling=bar` 下加载；新训练必须用 `tick`。  
旧模型在 tick 模式下行为无意义，需重训。
