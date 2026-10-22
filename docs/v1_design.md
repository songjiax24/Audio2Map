# Audio2Map v1 设计文档（唯一依据）

- 版本：草稿 v0.3，2026-10-22 (UTC+8)
- 适用对象：在仓库中实现 v1 的 Cursor
- 来源：(1) 设计决策记录（10-07 至 10-21，后定的覆盖先定的）；(2) 10-07 至 10-17 的架构讨论记录；(3) 仓库 `exports/` 下的分析结果（只读核对）。

## 修订记录

v0.3（2026-10-22，按 Cursor 实现前复述与用户决定修订）：

1. log-mel 改为固定参考：`x = (ln(mel + 1e-5) − μ) / σ`，不用 `ref=np.max` / `top_db`，不做逐曲响度归一化，不保存数据集统计文件；μ、σ 为写死在代码里的版本化常数；`fmin=20`（第 6.1 节）。
2. 红线近重复（`0 < Δ < EPS`）合法、不拒绝，由后一条红线生效；只有完全相同才拒绝；两类都要统计（第 2.6 节）。
3. 打标签以第 3.3/3.4/3.5/2.5 节正文为准；fallback 并列时先取较小分母，再取较小分子（第 3.4 节）。
4. 第 3.7、4.1、9 节中依赖统计脚本的数字一律只作参考，以正文规则为准（第 9 节）。
5. 仓库现状更新为 HEAD `767ae06`（第 11.1 节）。

v0.2（2026-10-21，按用户审阅意见修订）：

1. v0 兼容：先给 v0 最终状态打 git tag `v0-final`；之后 v1 可直接修改 v0 代码，不要求向后兼容（第 0、2.1、11 节）。
2. 时间单位统一为毫秒；`Δt = 256000/22050 ms ≈ 11.61 ms`（第 0、6.1 节）。
3. 重采样：ffmpeg 只按原始采样率解码，Python `soxr` 包以默认 HQ 档重采样到 22050 Hz（第 1.1 节，原 Q1 解决）。
4. 残拍内部格点也做重合判定：距红线不超过 EPS 的格点视为与红线重合并去掉（第 2.5、3.1 节）。
5. 音频缓冲区改为左右对称、全部取真实样本：`[(2g_lo−2)·128 − 512, 2g_hi·128 + 512)`，`center=False`（第 6.1 节）。
6. padding 条款改为说明：当前设计下自动满足（第 6.1 节）。
7. cross-attention 窗口中 floor 的含义补充说明（第 6.5 节）。
8. condition 17 维改为【待定】（第 6.6 节）；训练默认 plain CE，不做 masked CE，非法 token 概率质量的辅助惩罚列为【可选】（第 7.1 节）；数据增强改为【待定】（第 7.3 节）。
9. 模块布局只保留原则，具体由 Cursor 决定（第 11.3 节）。
10. 原 Q2：no-information、mixed、round 三类全部进训练，与 floor 谱用同一套打标签规则；依赖语料的基准数字改为“floor 参考值，需在扩大后的语料上重算”（第 1.3、1.5、3.4、9 节）。
11. 原 Q3–Q12 按用户意见写定（Q5 拒绝、Q6 沿用 10-18、Q7 简化、Q8 不算失败、Q9 简化为 `[G_start, G_end)`、Q10 按 `L_i`、Q11 导出不失败）。
12. 原 Q12 核对：按 exports 中的分档表，`(0, 1e-7]` 为 674 个（初稿误写 681），重合 section 为 6,717 个（初稿误写 6,724）（第 2.4、9 节、附录 B）。
13. 评测增加同轨重叠/过近的合理性指标（新增第 9.1 节）。

---

## 0. 目的与使用方式

1. 本文档是 v1 实现的**唯一依据**。旧的对话、旧的 plan、v0 文档与本文冲突时，以本文为准。开始实现前，先给 v0 的最终状态打 git tag `v0-final`，需要复现 v0 时 checkout 该 tag。之后 v1 可以直接修改 v0 代码，**不要求向后兼容**（见第 11 节）。
2. 每条规则标注状态：
   - **【已锁定】**：照做。若实现时发现矛盾或数据不符，停下来报告，不要自行修改规则。
   - **【待定】**：尚未决定。实现时写成可配置项或留出接口，不要自行选定。
   - **【可选】**：以后的消融或改进，v1 不做。
   - **【待确认】**：资料之间有冲突或缺失，需要用户确认。v0.2 中已无此类条目；新出现的写进 `open_questions.md`。
3. 时间单位：全项目时间一律为**毫秒（float64）**，包括音频 token 间隔 `Δt = 256000/22050 ms ≈ 11.61 ms`。不使用秒。
4. 唯一容差常数：`EPS = 1e-6`（ms）【已锁定，10-21】。窗口端点余量、红线重合判定和残拍内格点的重合判定都用它。fallback 并列判定另有 `TIE = 1e-12`（见 3.4）。

任务定义【已锁定，10-17】：

```
(Audio, Timing, Condition) → Chart        # osu!mania 4K
```

Timing（红线）是**输入条件**，不是模型的预测目标。训练时用谱面自带的 timing，不算信息泄漏。timing 从哪里来（现成谱面、人工测量、将来的自动测量）不属于 v1 范围。

---

## 1. 数据

### 1.1 音频解码【已锁定，10-18】

- **只用 ffmpeg 解码**，版本固定为 ffmpeg 7.1.1（conda-forge `gpl_hbbdf940_911`），把 `ffmpeg -version` 的第一行写进缓存元数据。不再用 v0 的“libsndfile 优先、ffmpeg 兜底”（`features/audio/loader.py` 不复用）。
- 以原始采样率解码为 `pcm_f32le`；混成单声道时**显式等权平均**各声道，不用 `-ac 1` 的默认行为。
- ffmpeg **只负责解码**，输出原始采样率的音频；再用 Python 的 `soxr` 包重采样到 22050 Hz，质量档位用默认的 **HQ**，`soxr` 版本固定并写进缓存元数据。输出长度 `round(n_native × 22050 / sr)`。
  - 原因：本机 ffmpeg 只带自带的 swresample，编译时没有带 libsoxr；用固定版本的 `soxr` 包，重采样结果不受 ffmpeg 编译方式影响，换机器也一致。
- `T_audio = len(y) / sr × 1000`（ms），取自生成 mel 的同一次解码结果，**不取整**，不用 `sf.info`。（v0 的 `audio_duration_ms` 会取整，不复用。）
- ffmpeg 会按 LAME 头裁掉编码延迟（delay+529）。接受这一行为。
- 音频库现状（10-18）：mp3 4,605 个、ogg 1,540 个，全部 6,145 个文件都能用 ffmpeg 7.1.1 解码。

### 1.2 音频拒绝规则【已锁定，10-18】

- 只有一种情况拒绝音频：**丢包导致时间线错位**，即容器时间戳继续前进，但解码输出把后面的样本提前拼接。检测必须**同时满足**：(1) 有被丢弃或无效的包；(2) 时间戳超前。只看“时间戳超前 N 个样本”会误杀 Bounce。
- 不用静音去填补丢包空洞。训练和推理用同一规则；推理时直接告诉用户“音频损坏”。
- 已排除：Florn (set 2322253)、Awakening Promise (1743718)、Cyclic Universe Theory (2298998)。
- 保留：Bounce (2080086)，在 216.078 s 处有 448 样本（10.16 ms）的容器空隙，但没有无效包，各解码器结果一致。
- 与 BASS 解码结果有差异的 33 个文件 / 112 张谱：保留，并记录在案。

### 1.3 谱面语料与分类【已锁定，10-19】

按“音符时间的取整方式”重新分类（只用可区分的端点：离简单拍点 d=1–4 不超过 1 ms、N≥10、占比 ≥0.9），19,267 张谱的结果（19,267 按实际处理结果理解，来历见表下）：

| 类别 | 数量 | 说明 | 是否进 v1 训练 |
|---|---:|---|---|
| floor（向下取整） | 17,132 | 端点共 27,900,588 个 | **进**【已锁定】 |
| no-information | 2,104 | 端点几乎都在整数毫秒上，无法判断 | **进**【已锁定，10-21】 |
| mixed | 28 | 明细见 `exports/mixed_fr/summary.tsv` | **进**【已锁定，10-21】 |
| round（四舍五入） | 3 | 5584325、5584332、351852 | **进**【已锁定，10-21】 |
| ceil | 0 | — | — |

- 19,267 这个基数的来历：10-18 记录为 ffmpeg 解码后保留 19,272 张（19,273 中只去掉 Florn）；再按 1.2 排除 Awakening Promise 的 4 张和 Cyclic Universe Theory 的 1 张后，正好是 19,267。按实际处理结果理解，不做特别处理。
- 四类谱面（共 19,267 张，再经 1.4 结构过滤）全部进训练，**统一**按 floor 处理：文件时间 `orig` 对应的真实时刻视为落在 `[orig, orig+1)` 内，用同一套窗口与 V80 打标签规则（第 3 节）。没有按类别区分的处理路径。
- 依据：3 张 round 谱已经在这套统一方案下测过，全部端点误差 ≤2 ms（最大 1.625 ms），见 `exports/round_charts_v80_formal/`。no-information 谱的端点几乎都在整数毫秒上，取整方向不影响结果。mixed 谱的命中率建议单独统计一次。
- 本文中凡是依赖语料的数字（谱面总数、端点数、split 数量、V80 命中率、fallback 数、格点计数等），都是**只在 17,132 张 floor 谱上算出的参考值**，需要在扩大后的语料上重算（第 9 节）。

### 1.4 结构过滤【已锁定，10-17/10-18】

- v1 **不复用** v0 的 `check_beatmap_eligibility`。不因变速、拍号变化或拍号 ≠4 而丢谱（`VARIABLE_BPM`、`VARIABLE_METER`、`METER_NOT_4` 都不用）。
- 只收 mania 4K 谱（沿用 `osu/parser.py` 的 `is_mania_4k`）。
- 任何**原始**端点（tap、hold 头、hold 尾）不在 `[0, T_audio)` 内 → 整张谱拒绝。
- 打标签后若出现结构冲突（同一格点、同一轨有两个端点；hold 头尾落在同一格点；等等）→ 整张谱拒绝。**绝不丢弃单个端点**，因为丢掉端点会产生非法序列。
- hold 跨过红线：不额外过滤。
- 两条红线时间完全相同 → 整张谱拒绝（见 2.6）。
- 统计并报告每一种拒绝原因的数量。

### 1.5 划分（split）【已锁定】

- 使用仓库已有的 `beatmapset_v1`（`audio2map/dataset/split.py`）：按 beatmapset 目录（`raw/` 下的父目录）哈希，seed 0，80/10/10。
- 17,132 张 floor 谱上的结果（floor 参考值）：train 13,691 / val 1,815 / test 1,626。
- 清单文件应在 `$AUDIO2MAP_DATA_ROOT/splits/beatmapset_v1.json`，但目前在数据盘和本机都没找到；原文件可能在远程机器上，也可能是在全量谱面上分的。划分只取决于 beatmapset 目录的哈希和 seed 0，因此重新生成的结果是确定的：直接重新生成，先筛出 17,132 张 floor 谱核对上面三个数字，再报告扩大后语料的三份数量。

---

## 2. BeatGrid【已锁定，10-17 / 10-18 / 10-21】

### 2.1 基本原则

- BeatGrid 是**唯一的时间来源**，用 float64 计算（与 osu! lazer 一致），每张谱只生成一次。
- 下游一律用索引 `(section, k)`（拍）和 `(section, k, 格点)` 引用位置，**不做浮点相等比较**。
- 不使用 `Fraction` / `Decimal` 精确有理数（已废弃，见附录 A）。
- v1 另写解析：直接读 `.osu` 原文的 `[TimingPoints]`，offset 与 beatLength 按 float64 解析。v0 的 `parse_timing_points` 会对 offset 做 `int(float(...))` 截断；可以直接把它改为 float64 解析（不要求兼容 v0）。

### 2.2 红线与 section

- 只用 uninherited timing point（红线，`uninherited=1`）。绿线（inherited）**全部忽略**：不产生 section，不进入模型，导出时也不写（第 8 节）。
- 第 i 个 section 从红线 `s_i` 开始，拍长 `L_i`（beatLength，ms）。拍首为 `a_{i,k} = s_i + k·L_i`。
- section i 的拍首必须**严格早于**下一条红线 `s_{i+1}`（考虑 2.4 的重合判定）。
- 拍号（meter）不使用；`<BAR>` 已删除。
- 红线 beatLength ≤ 0 或非有限值 → 拒绝整张谱，并单独统计【已锁定，10-21】。

### 2.3 首条红线之前：向前延伸（方案 A）

- 首条红线 `s_0` 之前，按 `L_0` 向前延伸拍首：`a_{0,k} = s_0 + k·L_0`，`k < 0`。理由：osu! 编辑器在首条红线前也用首条红线的节拍器。
- 拍序号 `b` 在延伸部分为负数，并跨过 `s_0` 连续计数。φ/ψ 在 `s_0` 之前同样计算（不再“前奏不加 phase”）。
- 网格范围：`G_start` = 不晚于 0 的最后一个拍首（可以为负）；`G_end` = 不早于 `T_audio` 的第一个拍首（拍首含红线）。按定义总有 `G_start ≤ 0`、`G_end ≥ T_audio`。网格层范围为 `[G_start, G_end)`；音符只能落在音频层 `[0, T_audio)`。

### 2.4 红线重合判定

对每条后续红线 `t_red = s_{i+1}`，在旧 section i 中：

```
k* = round((t_red − s_i) / L_i)
若 |t_red − (s_i + k*·L_i)| ≤ EPS (= 1e-6 ms)：视为重合
```

- 重合：旧 section 的第 k* 拍首与红线视为同一点，**不留残拍，也不多出一个拍**。
- 最近拍号 `k*` **只**用于这一判定。
- 依据：红线到旧段最近拍首的距离在 17,132 张 floor 谱上的分布，在 `4e-8` 到 `2e-6` ms 之间一个 section 都没有，`1e-6` 落在这段空白里。重合（≤1e-6）共 6,717 个 section（恰好 0 的 6,043 个 + `(0, 1e-7]` 的 674 个；`(1e-7, 1e-6]` 为 0），见 `exports/floor_d64/vocab/all/v80_residual_bins/residual_decade_hist.tsv`（floor 参考值）。旧做法“先 floor 再检查”会产生 294 个假的近整拍残拍，已废弃。

### 2.5 残拍（截断并保留）

- 不重合时，从**不晚于 `t_red` 的最后一个拍首** `s_last` 起算，残拍长度 `r = t_red − s_last`。
- 残拍在红线处截断，**不论多短都保留**。不删除、不合并到红线，没有任何长度阈值（残拍本身是否存在只由 2.4 决定）。
- 残拍内部的格点同样做重合判定：时间 t 满足 `t ≥ t_red − EPS` 的格点视为与红线重合，去掉，由新 section 的 `0/1` 代表；只保留 `t < t_red − EPS` 的格点。用的是同一个 `EPS`。
- 依据：残拍长度在 `1e-2` ms 到 1 ms 之间平滑增长，1 ms 处没有断层，找不到客观阈值；ψ 已经让模型能看到残拍造成的额外 `<BEAT>`（见 6.4）。

### 2.6 同一时刻的两条红线

- 两条红线时间完全相同（Δ = 0）→ 含义不明确，**报错并拒绝整张谱**。
- 近重复（`0 < Δ < EPS`）→ 合法，**不拒绝**。按 2.4，后一条红线与前一条的第 0 拍首重合（`k* = 0`），前一条 section 不产生任何拍首或格点，实际由后一条红线生效；落在这段区间内的端点按正常的窗口 + V80 规则标到后一条红线的格点上。
- 两类情况各自涉及的谱面数和端点数都要统计，写进日志或报告；拒绝数计入语料统计。【10-22】
- lazer 的解码器会让文件中先出现的那条生效，但这只是代码副作用（`LegacyBeatmapDecoder.cs`），没有文档，stable 的行为也不明确，因此不作依据。当前语料中 0 例。

### 2.7 BeatGrid 应提供的接口（建议）

- 拍序列：每个拍首的 `(section, k)`、时间、全局拍序号 `b`、该拍的实际长度（普通拍 = L_i，残拍 = r）。
- 给定时间 t：所在拍、拍内相位 φ、全局拍序号 b。
- 给定 `(section, k, 位置 p)`：时间 `a_{i,k} + p·L_i`。
- `G_start`、`G_end`、`T_audio`、合法格点判定。

---

## 3. 位置词表 V80 与正式打标签【已锁定，10-20】

### 3.1 V80 词表

- 分母集合 `D = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 24, 32}`（因数封闭的 A′ 方案）。
- 位置为既约分数 `p = j/d`，`0 ≤ j < d`，`gcd(j, d) = 1`，共 **80** 个（`0/1` 记作 d=1）。
- 各分母格点数：1:1, 2:1, 3:2, 4:2, 5:4, 6:2, 7:6, 8:4, 9:6, 10:4, 12:4, 14:6, 16:8, 18:6, 24:8, 32:16，合计 80。
- 注：有一条早期记录漏写了 18；以本节和 `scripts/v80_formal_label_counts.py` 的 `DENS` 为准，含 18。
- 位置格点时间：`a_{i,k} + p·L_i`，`p ∈ [0, 1)`。残拍中只保留时间 `< t_red − EPS` 的格点（2.5）。
- 没有单独的慢速 BPM 词表。

### 3.2 打标签输入

- 每个端点（tap、hold 头、hold 尾）的文件时间 `orig` 为整数毫秒。四类谱面都按 floor 解释（1.3）。

### 3.3 窗口规则（主规则）

```
窗口 W = (orig − EPS, orig + 1 + EPS)        # 开区间，EPS = 1e-6
候选 = 窗口内的所有 V80 格点（跨 section、跨拍都算）
选择：最小既约分母 d；
      d 相同 → 离 orig 更近者；
      仍相同 → 较小的分子（k）。
```

### 3.4 fallback（窗口为空时）

- 在全部 V80 格点中取离 `orig` 最近者；距离差在 `TIE = 1e-12` 以内视为并列，先取较小的既约分母 d，再取较小的分子 k【10-22】。
- floor 参考值：窗口命中率 99.8206%（27,850,532），fallback 50,056 个端点；需在扩大后的语料上重算。

### 3.5 候选范围与音频范围

【已锁定，沿用较新的 10-18 决定】
- 候选只取合法格点：格点时间在 `[0, T_audio)` 内（且在网格层 `[G_start, G_end)` 内，残拍中满足 2.5）。
- 原始端点越界的谱已在 1.4 被整谱拒绝。若某端点在合法格点中找不到候选（窗口为空且 fallback 也没有合法格点），整谱拒绝并统计数量。
- 10-17 的“先吸附再检查范围”作废。正式 V80 统计脚本当时是否限定了这个范围未核实，重算基准时如有差异，列出原因。

### 3.6 格点归属 section

- 每个格点归属于它所在拍的 section；旧段格点不得延伸到下一条红线之后（含重合时的红线本身，红线属于新 section 的 `0/1`）。

### 3.7 红线边界结论【已锁定，10-21】

- 维持正式 V80 加截断残拍，**红线附近不做任何改标或合并**。
- 现象：离下一条红线 `(0, 1]` ms 的端点 9,491 个，其中 6,323 个的窗口内有残拍起点，全部标在旧段 `0/1` 上；另外 3,168 个没有残拍。离红线 `(1, 2]` ms 的端点 470 个，其中 390 个窗口内有残拍起点。
- 解释：编辑器里旧段拍点和红线几乎重合、无法分辨，音符落在哪一边取决于点击位置，大约 7,600 个端点（0.03%）多半是无意的。两种写法还原的时间相差不到 1 ms，在 osu! 的 2 ms 容差内（osu! 排行检查：偏离 ≥2 ms 警告，<1 ms 忽略）。
- 数据中找不到客观阈值，所以不处理。训练后在验证集上单独检查红线附近的生成结果。

### 3.8 打标签后的结构检查

- 同一 `(section, k, p)`、同一轨出现两个端点 → 整谱拒绝（只针对训练数据；生成结果的导出见第 8 节）。
- hold 尾必须严格晚于 hold 头（不同格点）。否则整谱拒绝。

---

## 4. Token 序列与文法

### 4.1 词表【已锁定】

| 组 | 数量 | 说明 |
|---|---:|---|
| `<PAD>` | 1 | 只用于 batch collate，不参与 loss，解码不会产生 |
| `<BOS>` | 1 | 只作前缀，不移动 cursor，不计 loss |
| `<EOS>` | 1 | 结束 |
| `<BEAT>` | 1 | 推进到下一个拍首 |
| `<POS_j/d>` | 80 | V80 位置 |
| `<ROW_abcd>` | 609 | 四轨联合事件状态 |

合计 693。id 的排列顺序由实现决定，但必须固定并写进 tokenizer 版本号。

- **没有** `<BAR>`、`<TIMING_POINT>`、initial `<ROW>`。
- ROW 609 的定义：每轨状态 ∈ {0 空, 1 tap, 2 hold 头, 3 hold 持续, 4 hold 尾}（与 v0 `LaneState` 相同）。全部 `5^4 = 625` 种中，去掉**没有任何事件**的 `2^4 = 16` 种（每轨只取 0 或 3），得 `625 − 16 = 609`。v0 词表里的 625 个 ROW 包括这 16 种非事件状态，v1 不包括。
- 在 17,132 张 floor 谱中 609 类全部出现，最少的 `ROW_2121` 出现 255 次，中位数 5,110（`exports/floor_d64/row_stats/summary.tsv`；floor 参考值，需在扩大后的语料上重算）。

### 4.2 序列形式

```
<BOS>  ( <BEAT> | <POS_p> <ROW_abcd> )*  <EOS>
```

- cursor 初始位于 `G_start` 所在拍的拍首（即包含时间 0 的拍）。`<BOS>` 不移动 cursor。
- `<BEAT>`：前进到 BeatGrid 中的下一个拍首（普通拍首或红线）。重合的红线只算一个拍首，只推进一次；残拍会产生一个很短的 `<BEAT>`，正常保留。
- 同一拍内，同一时刻所有轨的端点合成一个 `<POS_p> <ROW_abcd>`；同一拍内 `p` **严格递增**。
- 空拍只发 `<BEAT>`。cursor 到达 `G_end` 后只允许 `<EOS>`。
- 第一个 token 可以是 `<POS_p>` 或 `<BEAT>`；当前拍没有合法格点时必须是 `<BEAT>`。
- `<POS_p>` 只能取当前拍中的**合法格点**：位置时间在 `[0, T_audio)` 内，且（残拍中）早于 `t_red − EPS`。
- `<POS_p>` 之后必须紧跟一个合法 `<ROW>`（`is_legal_row` 规则见 v0 `tokens/vocab.py`，可复用）。

### 4.3 encode / decode

- encode：谱面 → BeatGrid → 每个端点按第 3 节打标签 → 按 `(拍, p)` 排序合并成 ROW → token 序列。
- decode：token + BeatGrid → 每个 `<POS_p>` 的时间 `t = a_{i,k} + p·L_i`（float64）→ 端点列表。
- 训练 teacher forcing 与逐步推理必须共用同一个状态机，并使用 observe **之前**的状态。

---

## 5. 解码约束【已锁定，10-18】

推理时用状态机做约束解码。训练用 plain CE，不加文法 mask（见 7.1）。

- (a) 只有在它**之后**还有合法格点时，才允许新的 hold 头。最后一个合法格点上不允许 hold 头。
- (b) 有 hold 未闭合时，只有当**下一拍**中还有合法格点（下一拍首 `< T_audio`）时才允许 `<BEAT>`。
- (c) 在最后一个合法格点上，ROW 必须把所有正在进行的 hold 收尾，且其他轨不得出现新的 hold 头。
- (d) 只有 cursor 到达 `G_end` 时才允许 `<EOS>`（与 4.2 一致）。(a)–(c) 已保证到达 `G_end` 时没有未闭合的 hold，因此 (d) 不再单独检查 hold；实现中保留一个 assert。
- 无死状态：在任何可达状态下至少有一个合法 token（此论证已锁定）。实现时用单元测试覆盖（尤其是残拍、`G_end`、hold 未闭合的组合）。
- 注：(a)–(c) 当初是按 1/48 网格写的；在 V80 下，“合法格点”指 `[0, T_audio)` 内、且未越过红线的 V80 格点。

推理失败原则【已锁定】：

- 输入本身有缺陷（音频损坏、重复红线等）→ 在开始时就检测并拒绝。
- 生成之后出现的模型侧缺陷（例如结尾有未闭合的 hold）→ 不算失败。由于 (a)–(c)，正常情况下不会出现未闭合 hold。若仍出现（说明实现有 bug），照常导出，在审计记录中报告并标记。

---

## 6. 模型结构

### 6.1 音频前端与 stem【已锁定】

- log-mel：22050 Hz、`n_fft=1024`、hop 128、128 mel、`fmin=20`，帧率约 172 Hz（沿用 v0 前端参数）。【已锁定，10-22】
- 取对数与归一化：`x = (ln(mel + 1e-5) − μ) / σ`。
  - 不用 v0 的 `power_to_db(ref=np.max, top_db=80)`：它按每首歌自身最大值做参考，丢掉了绝对响度，而且随切块位置变化，破坏分段与整首一致。
  - 不做逐曲响度归一化，不保存数据集统计文件。
  - μ、σ 是固定常数：在训练集上用固定 seed 抽样（如 50 首）统计 `ln(mel + 1e-5)` 的全局均值和标准差，算一次后写死在代码里，带版本号（如 `MEL_NORM_V1`），并记录计算脚本和抽样方式。做法类似 Whisper 的固定仿射 `(x + 4) / 4`。
  - 依据：50 首歌统计中 mel 值 72.5% 小于 1、中位数 0.247，`log1p` 会压扁这部分；`ln(mel + 1e-5)` 范围约 −11.5 到 8、中位数约 −1.4，stem 前没有 LayerNorm，因此需要固定仿射把输入调到大致零均值、单位方差（`exports/mel_energy_stats/`）。
- Stem（Whisper 风格）：`Conv1d(128→512, k=3, s=1, p=1)` + GELU，`Conv1d(512→512, k=3, s=2, p=1)` + GELU，输出约 86 Hz。
- 音频 token 间隔 `Δt = 256000/22050 ms ≈ 11.61 ms`；全局 token 下标 `g` 的中心时间 `t(g) = g·Δt`，`g=0` 对应物理时间 0。
- stem 之后**不加 LayerNorm**，**不用** Whisper 的绝对位置编码。
- v0 的 `compute_tick_grid_features`（按 tick 插值对齐）不复用。

全局 lattice 对齐（10-11 定稿，沿用）：

- 编码覆盖的时间范围 `[t_lo, t_hi)`；`g_lo = ceil(t_lo/Δt)`，`g_hi = ceil(t_hi/Δt)`，可见 token `g ∈ [g_lo, g_hi)`，`S = g_hi − g_lo`。当 `|t/Δt − 最近整数| ≤ FRAME_INDEX_SNAP`（`1e-8`，定义在 `features/audio/v1/constants.py`）时取该整数，避免二进制误差把正好落在帧边界上的时间推进下一帧。代码里这个下标叫 `frame_index`（`frame_lo` / `frame_hi`），是 mel 帧，不是拍范围。拍范围在代码里是 `coverage_start_ms` / `coverage_end_ms`，对应本节公式里的 `G_start` / `G_end`。公式符号 `g` 仍指 mel 帧。
- 需要的全局 mel 帧为闭区间 `[2g_lo − 2, 2g_hi]`；帧 i 的中心样本为 `i·128`。
- 缓冲区取样本区间 `[(2g_lo − 2)·128 − 512, 2g_hi·128 + 512)`，长度 `256S + 1280`。左右各多出 512 个样本（半个 STFT 窗口），使第一帧和最后一帧的整个窗口都是真实样本。样本落在 `[0, N)` 内时拷贝 PCM，否则写 0（这就是歌曲时间轴上的静音）。
- STFT 用 `center=False`：帧数 `1 + (256S + 1280 − 1024)/128 = 2S + 3`，正好是所需的帧，不需要再截掉。第 j 帧的中心样本为 `(2g_lo − 2 + j)·128`。
- stem 输出长度 `S+2`（第二层 k=3、s=2、p=1：`floor((2S+3 + 2 − 3)/2) + 1 = S+2`），保留 `q = 1..S`，对应 `g = g_lo + q − 1`。
- 修订说明：初稿的缓冲区从 `(2g_lo − 2)·128` 开始、长度 `256S + 768`，并用 `center=True`。这样左边半个窗口靠 librosa 自己补值，不是真实 PCM，在歌曲中间分段时第一帧与整首计算的结果不一致，已废弃。
- 必须有测试：同一首歌分段编码与整首一次编码，同一帧的 stem 在 `atol=1e-6` 内相等。不能要求逐位相等：两段缓冲长度不同，float32 的 mel 滤波器和 Conv1d 矩阵乘不可结合，内核还可能随张量形状变化。CI 上见过约 `5.96e-8`（`2**-24`）的差。
- 三种“零”必须区分：歌曲时间轴上的静音（有效 key）；STFT/卷积的 guard（不进 encoder）；batch padding（`audio_mask=False`，不进 stem，不分配 g）。
- 编码覆盖范围 `[t_lo, t_hi) = [G_start, G_end)`。原定稿为 `[min(0, s_0), T_grid)`；按 2.3 的定义 `G_start ≤ 0` 总成立，所以不再需要 `min(0, ·)`。
- padding 必须等于真实静音的特征值（v0 用 0 填 log-mel 视为 bug）。当前设计下这一条**自动满足**：歌曲范围外的部分在 PCM 层写 0，算出来就是真实静音的特征；batch padding 被 mask 掉。只有以后在特征层补值时（例如切块训练）才需要注意，并区分训练裁剪边界与真实歌曲边界。

### 6.2 Encoder【已锁定】

- 8 层 Transformer encoder，`d_model = 512`，8 head，head 维度 64，FFN 为 SwiGLU、隐藏维 1536。
- 双向局部注意力：对裁剪后下标 e，窗口 `[e−512, e+512) ∩ [0, S)`，内部正好 1024 个 key。
- RoPE 位置 = 全局音频 token 下标 g（窗口滑动时不重置）。
- Encoder **不看** condition，也**不看** φ/ψ。

### 6.3 Decoder【已锁定】

- 8 层，`d_model = 512`，8 head，每层顺序：局部 causal self-attention → cross-attention → SwiGLU FFN（1536）。
- Self-attention 窗口：`0 ≤ q − k < 1024`（token 下标）；RoPE 位置 = decoder token 下标。没有 learned position embedding。
- token embedding：POS 与 ROW 都用普通的可学习 embedding【已锁定】；结构化 embedding 是以后的消融【可选】。
- 输出 LM head 默认与 token embedding 共享权重（沿用 v0 做法，不是架构约束）【可选】。

### 6.4 φ/ψ 拍相位注入【已锁定，10-17】

- 每个可见音频 token g，在其中心时间 `t(g)` 处计算：
  - 拍内相位 `φ ∈ [0, 1)`（在该时刻所在拍内，按该拍所属 section 的 `L_i` 计；残拍中同样按 `L_i` 计算）；
  - 全局拍序号 `b`：跨红线连续计数，不按 section 重置，不取模 4；首条红线前为负数；
  - 双拍奇偶相位 `ψ = (b + φ) / 2`。
- 特征：φ 取 Fourier `k = 1..8` 的 sin/cos（16 维）；ψ 取 `k = 1` 的 sin/cos（2 维）。拼接后经**同一个**线性层 `W_φ` 加到 encoder 输出上：`H_context = H_audio + W_φ·[f(φ); f(ψ)]`，作为 cross-attention 的 K/V。不新增其他通路。
- 作用：相邻 token 之间经过 0、1、2 个拍首都能区分（只有 1 与 3 会混淆；语料中相邻 token 间拍首数最多为 2，共 289 处）。残拍产生的额外 `<BEAT>` 因此对模型可见。
- 依据统计范围是 19,278 张可解析的 4K 谱。
- 残拍内的 φ 按 `L_i` 归一：`φ = (t − a_{i,k}) / L_i`，不按残拍长度归一【已锁定，10-21】。

### 6.5 Cross-attention【已锁定】

- query 位置 `p_q = cursor_time / Δt`（可为小数、可为负）。cursor_time 取预测当前 token **之前**的状态（预测 `<POS_p>` 时不能用尚未生成的 p；预测其后的 `<ROW>` 时可用已确定的位置）。
- key 选择：`k ∈ [floor(p_q) − 128, floor(p_q) + 384) ∩ [g_lo, g_hi)`，最多 512 个，边缘可以更短，不补假 key。
- 用 floor 的含义：`floor(p_q)` 是不晚于 query 的最后一个音频 token；“过去 128”包含它，“未来 384”严格在它之后。这是一个约定。
- RoPE：query 用小数 `p_q`，key 用全局整数 g；`floor` 只用于选窗口，不用于 RoPE。
- 对照方案：512/512 对称窗口【可选】。
- RoPE base θ = 10000（标准值）。是否调整列为【待定】。

### 6.6 Condition 与 AdaLN-Zero【已锁定】

- 条件向量 x 的具体字段【待定】。当前候选：v0 `features/cond/vec.py` 的 18 维去掉 `canonical_bpm_norm`，剩 17 维：`osu_sr_norm`、8 个 analyzer 比例、8 个 etterna 指标（顺序与 v0 `COND_VEC_NAMES[:17]` 相同，归一化沿用 `build_cond_vec`）。实现时字段列表写成可配置，下面的维数随字段数变化。
- 缺失值：`z = [x ⊙ m; m]`（m 为可用性 mask），按当前候选共 34 维；MLP `34 → 256 → 256` 得到 `c ∈ R^256`。不用 learned missing embedding，不做字段级拆分。
- v0 缺字段时直接抛 `CondVecError`；v1 要改为用 mask 表示。
- 每个 decoder 子层（SA、CA、FFN）各自由 c 产生 `γ, β, α`；`α` 零初始化（AdaLN-Zero）。条件因此会影响 cross-attention 的 query。
- Condition **不进入** audio encoder。
- 训练时模拟条件缺失（mask 的分布）属于训练协议【待定】。

---

## 7. 训练

### 7.1 Loss

- next-token 交叉熵【已锁定】。
- 训练默认 plain CE【已锁定】；**不做** masked CE（训练时对非法 token 做 mask）。原因：被 mask 的 token 得不到梯度，模型学不会压低非法 token，只能依赖解码约束，训练与推理不一致；真实标签全部合法，plain CE 已隐式学习文法。
- 辅助损失【可选】：对分给非法 token 的概率质量加惩罚项。仅当观察到解码时文法约束频繁被触发时再考虑。
- `<BOS>` 只作前缀，loss 从第一个谱面 token 开始；`<PAD>` 不计 loss。

### 7.2 硬件【已锁定】

- 训练用 RTX 5090；RTX 3070 只用于推理。

### 7.3 待定事项

- 训练样本方式：整首歌 / 切块 / 滑动窗口【待定，写数据管线前必须决定】。BeatGrid、打标签、tokenizer 与此无关，可以先实现。
- Cross-attention 的训练显存策略（不分块、按层重算、FlexAttention 等）【待定，先做 benchmark】。benchmark 需覆盖长音频、不同 token 长度与 batch size，测前向和反向的峰值显存与单步时间；推理时每层预先投影整段 K/V，再按窗口 gather（已定）。
- 数据增强方案【待定，可最后决定】：候选包括变速，以及 4K 左右镜像（把 ROW 的轨道顺序反过来，实现简单）等。
- Decoder 1024 窗口与 8/8 层数是否足够【待定，看实验】。
- 优化器、学习率、batch 大小等资料里没有，属于实现时的普通配置。

---

## 8. 导出【已锁定】

- 只写红线（原样照抄输入 timing 的红线），**不写绿线**。
- 端点时间：float64 网格时间向下取整（floor）为整数毫秒。不做精确有理数导出，不反推 BPM 修正。与原谱逐毫秒一致不是目标。
- 正式生成的导出**不失败**【已锁定，10-21】。取整后若出现同一轨同一毫秒的重复 note，或间隔很短的同轨事件（只可能发生在红线前不到 1 ms 的残拍附近），照常导出，**不删除、不移动、不修改** note，只在审计记录里写一条警告。osu! 可以打开和游玩这样的谱面，只是谱面不合理。hold 尾取整后不晚于头的情况同样记警告。（10-11 的“整次导出失败”作废。）
- 注意 v0 `osu/export.py` 的 `filter_notes_for_export` 会静默丢事件，v1 导出不得这样做。

---

## 9. 验证基准数字

下列数字都是**只在 17,132 张 floor 谱上算出的参考值**。实现后分两步：(1) 先在这 17,132 张 floor 谱上复现，用于核对实现是否正确；(2) 再在扩大后的语料（四类全部，见 1.3）上重算，作为 v1 的新基准。不一致时列出差异与原因，**不要为了对上数字悄悄改规则**。下表“性质”一栏指第 (1) 步的要求。实现以正文规则（第 2.5、3.3–3.5 节）为准，统计脚本的数字（包括第 3.7 节端点数、第 4.1 节 ROW 计数）都只作参考值；若复现不一致，解释原因即可，不改规则去对齐脚本。【10-22】

| 项目 | 参考值 | 来源 | 性质 |
|---|---|---|---|
| floor 谱数 / 端点数 | 17,132 / 27,900,588 | `v80_formal_label_counts/summary.tsv` | 硬基准 |
| 窗口命中 | 27,850,532（99.8206%） | 同上 | 硬基准 |
| fallback | 50,056 | 同上；与 `fallback_err/summary.tsv` V80 行一致 | 硬基准 |
| 最少格点 | 23/32，552 次 | 同上 | 硬基准 |
| 最多格点 | 0/1，12,928,026 次 | 同上 | 硬基准 |
| 格点计数中位数 | 3,834 | 同上 | 硬基准 |
| 计数 <1000 的格点 | 10 个（都是 d=32） | 同上 | 硬基准 |
| 各分母合计 | 见 `by_denominator.tsv` | 同上 | 硬基准 |
| 全部端点误差 ≤2 ms 的谱 | 16,798 张（98.05%） | `fallback_err/summary.tsv` V80 | 硬基准 |
| 误差 >2 ms 的端点 | 12,751 | 同上 | 硬基准 |
| 重合 section（≤1e-6） | 6,717 | `v80_residual_bins/residual_decade_hist.tsv`（6,043 + 674） | 硬基准（6,717 + 93,586 = 100,303，与该表 section 总数一致） |
| 残拍总数 | 93,586 | `v80_residual_bins/summary.tsv` | 参考值（当时用 1e-7 判定重合，与 1e-6 结果相同） |
| `[0.01, 1)` ms 残拍 | 13,876（1,092 张谱） | 同上 | 参考值 |
| ROW 类数 / 最少 / 中位数 | 609 / 255 / 5,110 | `row_stats/summary.tsv` | 参考值（该统计的打标签口径未核实） |
| split | 13,691 / 1,815 / 1,626 | 记录 | 硬基准（重新生成后核对，见 1.5） |

另外需要报告：

- encode → decode 往返：端点 `(section, k, p)` 一致率；导出时间与原 `orig` 的误差分布（max、median），以及误差 ≤2 ms 的谱面比例。
- 各拒绝原因的数量（第 1.4 节）。
- 重合判定的单元测试，包括 Seychelles（set 2425634，beatLength `165.9999`）和 Nokia Tune 2007（set 2456432，红线 `18001.235951`，残拍约 `4.25e-5` ms）这类非常短但大于 EPS 的残拍：应保留为残拍。

另有一条旧记录写“103,693 个 section 中 7,605 个重合（与 Fraction 精确计算一致）”。它的 section 总数与上面的 100,303 不同，应是统计范围不同的旧结果，不作基准。

### 9.1 生成结果的评测要求【已锁定，10-21】

- 解析生成谱面时，必须能接受同一轨同一毫秒的重复 note 以及间隔很短的同轨事件：不报错，也不静默去重。
- 合理性指标，阈值沿用 lazer 谱面检查器 `CheckManiaConcurrentObjects`，只比较同一轨上相邻的两个物件（prev 在前，next 在后；tap 的结束时间等于开始时间）：
  - 重叠（concurrent）：`next.start ≤ prev.end + 2 ms`。包括落在 hold 期间的 note。
  - 过近（almost concurrent）：`|next.start − prev.end| < 10 ms`。
  - 两项分别统计次数（以及出现的谱面数）。
- 参考：osu!mania ranking criteria 规定“Two notes cannot be placed in a single column at the same timestamp”（https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania）。
- v0 的网格上不会出现这类情况；v1 因为有残拍，可能出现。游戏内 stable / lazer 对这类情况如何判定未查证，不作结论。

---

## 10. 待定事项

待定（不影响 BeatGrid / 打标签 / tokenizer 的实现）：

1. 训练样本方式（整首 / 切块 / 滑动窗口）。
2. Cross-attention 训练显存策略（benchmark 后定）。
3. 数据增强方案（变速、左右镜像等）。
4. RoPE base、condition 字段与 mask 分布、decoder 窗口与层数是否足够。
5. 是否加入非法 token 概率质量的辅助惩罚（【可选】，不做 masked CE）。
6. 推理采样策略（资料中没有）。

原 Q1–Q12 已全部解决。剩余事项见 `open_questions.md`。

---

## 11. 仓库现状与模块布局原则

### 11.1 现状（2026-10-21 只读检查）

- 分支 `master`，最新提交 `767ae06`（`.gitignore` 加入 `exports/`；其前一个为 `3440698` v0 baseline 报告）。实现前先在此提交打 tag `v0-final`。仓库已重构为分层包，依赖方向由 import-linter 强制：
  `cli → eval → train → generate → dataset → model → features → metrics → tokens → grid → osu → utils`。
- **v1 已有的部分**（未提交）：红线、BeatGrid、V80 打标签、v1 tokenizer、ffmpeg+soxr 解码与 log-mel、`model/audio_stem.py`。数据集窗口、训练、导出、条件向量和 8 层模型还没有。`eval/v1_backend.py` 仍是 v0 评测适配器壳子，不是 v1 tokenizer。
- 分析脚本（原 WSL `/tmp/a2m_*.py`，共 24 个）已复制到仓库目录 `scripts/analysis/`，导入路径已改为本地目录。它们与 `scripts/v80_formal_label_counts.py` 都不进 git（写在 `.git/info/exclude`）；`exports/` 已写入 `.gitignore`（提交 `767ae06`）。这些脚本只用于核对数字，不是 v1 正式代码。另有一部分旧脚本在 `C:\Users\32279\a2m_*.py`。
- `docs/` 只有 `ARCHITECTURE.md`、`chart_tokens.md`（v0 token 说明）、`data_collection.md`。本文件建议放在 `docs/v1_design.md`。

### 11.2 可复用

| 模块 | 用途 | 说明 |
|---|---|---|
| `osu/parser.py` | 读 `.osu` 分节、`is_mania_4k`、`parse_notes`、`x_to_column` | v0 的 `parse_timing_points` / `TimingPoint` 保持原样。v1 红线在 `osu/v1/timing.py`，类型是 `Redline`，不是 `TimingPoint` |
| `tokens/vocab.py` | `LaneState`、`is_legal_row`、`apply_row`、`is_event_row` | 可复用；v1 词表（693）的组织方式由实现决定 |
| `features/cond/vec.py` | 条件字段与归一化 | v1 去掉 BPM 维并加 mask |
| `dataset/split.py` | `beatmapset_v1` 划分 | 直接用 |
| `model/rotary.py` | RoPE | 需支持小数位置 |
| `osu/export.py` | 写 `.osu` 的外壳 | 时间与红线规则按第 8 节，注意 `filter_notes_for_export` 会静默丢事件 |

v1 不沿用其逻辑（可以直接改写或替换）：`grid/timing.py`（`CanonicalTiming`、`ms_to_tick`）、`dataset/filter.py`（`check_beatmap_eligibility`）、`features/audio/loader.py`、`features/audio/tick_features.py` 的对齐、`tokens/state.py`（`ChartState` 以小节为单位）、`dataset/windows.py`（16 小节窗口）。

### 11.3 模块布局

过渡期的 v1 代码放在子包里，不使用 `timing_v1.py`、`v1_frontend.py` 这类半前缀。v0 的 `parse_timing_points` 和 `features/audio/loader.py` 保持原样。

| 路径 | 职责 |
|---|---|
| `audio2map/utils/errors.py` | `ChartRejected` 与一一对应的拒绝原因 |
| `audio2map/osu/v1/timing.py` | float64 红线（`Redline`，不是 v0 `TimingPoint`）；`offset` 与 `beatLength` 必填，缺 `meter` 用 4 |
| `audio2map/utils/tolerance.py` | `EPS`、`TIE` 只定义这一处 |
| `audio2map/grid/v1/beatgrid.py` | BeatGrid（第 2 节）。拍范围是 `coverage_start_ms` / `coverage_end_ms` |
| `audio2map/grid/v1/positions.py` | `V80_DENS` 与 80 个格子 |
| `audio2map/tokens/errors.py` | `IllegalTokenError` |
| `audio2map/tokens/v1/labels.py` | V80 窗口打标签（第 3 节） |
| `audio2map/tokens/v1/vocab.py` | v1 词表（693） |
| `audio2map/tokens/v1/state.py` | 状态机（`Cursor.beat_id`）与解码约束（第 4、5 节） |
| `audio2map/tokens/v1/structure.py` | 端点、音频范围、重复格、hold 顺序 |
| `audio2map/tokens/v1/codec.py` | `encode_chart` / `decode_tokens` |
| `audio2map/features/audio/v1/` | ffmpeg+ffprobe 解码、拒绝检测、重采样、log-mel。帧下标是 `frame_index` |
| `audio2map/model/audio_stem.py` | stem 卷积（`nn.Module`），`STEM_CHANNELS` 在这里 |
| `tests/` 下与包同构的 `v1/` | 单元测试 |

还没写、本轮不建：`beat_phase.py`、`cond/v1_vec.py`、`dataset/v1_filter.py`、8 层 `model`、`generate/v1_*.py`。

原则：

- 遵守现有分层，依赖方向由 import-linter 强制，不得新增反向依赖。
- `EPS` 在全项目只定义一处。
- 单元测试放在 `tests/` 下，覆盖第 2、3、5 节的边界情况和第 6.1 节的分段一致性。

---

## 附录 A：已废弃方案

| 方案 | 废弃时间 | 原因 / 替代 |
|---|---|---|
| `<TIMING_POINT>` token 与 `E_tp` 红线嵌入 | 10-17 | timing 只作输入；改用 φ/ψ 与 `<BEAT>` 推进到红线 |
| 查询函数 r(t) 注入 decoder 的 timing 条件、稀疏 timing 事件 token | 10-17 | 同上 |
| `Fraction` / `Decimal` 精确时间 | 10-17 | 改为 float64（跟随 lazer）+ 索引引用 |
| `<BAR>` 与拍号、initial `<ROW>` | 10-08 起 | 删除 |
| 1/48 网格 `<POS_0..47>` | 10-20 | 不能表示 1/5、1/7、1/9；改为 V80 |
| 1/48 下 AwayFromZero 取整、红线优先、并列取后者 | 10-20 | 改为 V80 窗口规则 |
| 等距取较早点、导出 `ROUND_HALF_UP` | 10-17/18 | 窗口规则；导出 floor |
| `ε = 1e-7` | 10-21 | 改为 `1e-6`（结果相同，统一常数） |
| “先 floor 再检查”的重合判定 | 10-18 | 产生 294 个假残拍；改为最近拍号判定 |
| 规则 P / M 之争 | 10-20 | 在 floor 语料上由正式 V80 窗口收口 |
| 规则 G（删除 <1 ms 残拍并入红线） | 10-21 | 找不到客观阈值；ψ 已使残拍可见 |
| 按分母改标到下一红线 | 10-20 | 被否决 |
| 首条红线前不加 phase（方案 B） | 10-17 | 改为向前延伸（方案 A） |
| libsndfile 优先、ffmpeg 兜底 | 10-18 | 只用 ffmpeg |
| v0 `check_beatmap_eligibility` 与变速/拍号过滤 | 10-17 | 不复用 |
| 用 0 填充 log-mel | — | 视为 bug；用真实静音值 |
| 慢速 BPM 专用词表、round 谱专用路径 | 10-20 | 不做 |
| 保留 `canonical_bpm_norm` 条件维 | 10-11 | 删除 |
| 6/6 层 | 10-11 起 | 改为 8/8 |
| 音频缓冲区 `p_buf = (2g_lo−2)·128`、长度 `256S+768`、`center=True` | 10-21 | 左边半窗不是真实 PCM；改为对称缓冲区、`center=False` |
| 导出取整冲突时整次导出失败 | 10-21 | 改为照常导出并记警告 |
| 只用 floor 谱训练 | 10-21 | 四类谱面全部进训练 |
| “不改 v0、另写 v1 模块” | 10-21 | 打 `v0-final` tag 后可直接修改 |
| GELU 2048 FFN | 10-15 | 改为 SwiGLU 1536 |

## 附录 B：关键分析结论与出处

| 结论 | 数字 | 路径（仓库 `exports/`） |
|---|---|---|
| 词表扫描：全部端点误差 ≤2 ms 的谱比例 | V34 91.6%、V40 95.1%、V48 95.7%、V58 96.8%、V64 97.1%、V80 98.1%、V96 98.6%、V144 99.2%、原 1/48 88.4% | `floor_d64/vocab/all/fallback_err/summary.tsv` |
| V80 正式标签计数 | 见第 9 节 | `floor_d64/vocab/all/v80_formal_label_counts/` |
| 红线边界冲突 | 8,075 个端点、746 张谱；A 1 / B 7,942 / C 132；窗口 8,027、fallback 48；旧段 `0/1` 6,367 | `floor_d64/vocab/all/v80_redline_boundary/summary.tsv` |
| 规则 G 模拟（有实现 bug，作废） | ε=1 时 6,210 个变成 31/32，≤2 ms 谱从 98.05% 降到 94.74% | `floor_d64/vocab/all/v80_residual_merge/summary.tsv` |
| 残拍长度分档 | `(0,0.01)` 395；`[0.01,1)` 13,876；`[1,2]` 768；`(2,5]` 638；`(5,10]` 584；`>10` 77,325 | `floor_d64/vocab/all/v80_residual_bins/` |
| 残拍起点上的端点（正式窗口） | `[0.01,1)` 6,060；`[1,2]` 456 | 同上 |
| 红线到旧段最近拍首距离分档 | 0：6,043；`(0,1e-7]` 674；`(1e-7,1e-6]` 0；`(1e-6,1e-3]` 211；`(1e-3,1e-2]` 614；`(1e-2,1e-1]` 6,900；`(1e-1,1]` 53,006；`(1,10]` 3,706；`>10` 29,149；合计 100,303 | `floor_d64/vocab/all/v80_residual_bins/residual_decade_hist.tsv` |
| round 谱 | 全部端点 ≤2 ms，最大 1.625 ms | `round_charts_v80_formal/` |
| mixed 谱明细 | 28 张 | `mixed_fr/summary.tsv` |
| ROW 统计 | 609 类，最少 255，中位 5,110 | `floor_d64/row_stats/summary.tsv` |
