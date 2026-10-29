# v1 设计文档：剩余事项（2026-10-22，v0.3）

原 Q1–Q12 已全部按用户意见写进 `v1_design.md`，不再列出。

剩余的只有【待定】实验项和实现时需要核对的事项，不需要用户现在决定：

| # | 事项 | 性质 | 位置 |
|---|---|---|---|
| R1 | 训练样本：每张训练谱都是整首 teacher forcing（方案 A）。不使用 C/D | 已锁定，2026-10-29 | 7.3 |
| R2 | cross-attention 训练显存策略 | 待定，先做 benchmark | 7.3 |
| R3 | 数据增强：镜像 p=0.5；变速 p=0.2，`Uniform[1, 1.5)`；soxr 时钟伸缩并对齐时间轴。不接 Mug 的 interval-mirror、随机列置换、独立 pitch | 已锁定，2026-10-29 | 7.3 |
| R4 | condition VERSION=3：official→rework→8 MSD→ln→hb→6 pattern；Rework 固定 OD=8；Pattern 为 FoundPattern 区间并集 / Duration（方案 A，analyze-once）；Composer mask | 已决，2026-10-23 | 6.6 |
| R5 | 是否加非法 token 概率质量的辅助惩罚（不做 masked CE） | 可选实验 | 7.1 |
| R6 | RoPE base、decoder 窗口与 8/8 层数是否足够 | 待定实验 | 6.5、7.3 |
| R7 | 推理采样策略 | 资料中没有 | 10 |
| R8 | 扩大语料（四类全部）后重算所有基准数字与 split 数量 | 实现时完成 | 1.3、1.5、9 |
| R9 | 正式 V80 统计脚本当时是否把候选限定在 `[0, T_audio)`；重算时若有差异需解释 | 实现时核对 | 3.5 |
| R10 | 2.4 中“`4e-8` 到 `2e-6` 之间没有 section”这一说法：exports 的分档表只能确认 `(1e-7, 1e-6]` 为 0，`(1e-8, 1e-7]` 有 2 个；更细的边界来自 10-21 的讨论记录 | 不影响 EPS=1e-6 的选择，可在实现时顺带核对 | 2.4 |
| R11 | 倍速：official SR 用 `set_clock_rate`，MSD 用 `music_rate`，Rework 用 `speed_rate`（时间除以 rate）。不加 rate 维。推理 rate=1 | 已锁定，2026-10-25 | 6.6 |
| R12 | Importance 是否替换六维 Pattern 的 Amount 时间份额 | 待办；当前否，cond 继续用方案 A 的 Amount | 6.6 |
