# v1 设计文档：剩余事项（2026-10-22，v0.3）

原 Q1–Q12 已全部按用户意见写进 `v1_design.md`，不再列出。

剩余的只有【待定】实验项和实现时需要核对的事项，不需要用户现在决定：

| # | 事项 | 性质 | 位置 |
|---|---|---|---|
| R1 | 训练样本方式：默认一张谱+对应音频为一条序列；启发序 A 整首 TF → C 长音频前向+窗内 loss+真实 decoder 前缀 → D 再加 audio halo | 暂定，2026-10-22；实验裁定绑 R2 与训练试点 | 7.3 |
| R2 | cross-attention 训练显存策略 | 待定，先做 benchmark | 7.3 |
| R3 | 数据增强（变速、4K 左右镜像等） | 待定 | 7.3 |
| R4 | condition VERSION=3：official→rework→8 MSD→ln→hb→6 pattern；Rework 固定 OD=8；Pattern 为 FoundPattern 区间并集 / Duration（方案 A，analyze-once）；Composer mask | 已决，2026-10-23 | 6.6 |
| R5 | 是否加非法 token 概率质量的辅助惩罚（不做 masked CE） | 可选实验 | 7.1 |
| R6 | RoPE base、decoder 窗口与 8/8 层数是否足够 | 待定实验 | 6.5、7.3 |
| R7 | 推理采样策略 | 资料中没有 | 10 |
| R8 | 扩大语料（四类全部）后重算所有基准数字与 split 数量 | 实现时完成 | 1.3、1.5、9 |
| R9 | 正式 V80 统计脚本当时是否把候选限定在 `[0, T_audio)`；重算时若有差异需解释 | 实现时核对 | 3.5 |
| R10 | 2.4 中“`4e-8` 到 `2e-6` 之间没有 section”这一说法：exports 的分档表只能确认 `(1e-7, 1e-6]` 为 0，`(1e-8, 1e-7]` 有 2 个；更细的边界来自 10-21 的讨论记录 | 不影响 EPS=1e-6 的选择，可在实现时顺带核对 | 2.4 |
| R11 | 倍速：`music_rate` / `set_clock_rate` / Rework `speed_rate` | 待办；cond 当前按 1.0 倍，未接 | 6.6 |
| R12 | Importance 是否替换六维 Pattern 的 Amount 时间份额 | 待办；当前否，cond 继续用方案 A 的 Amount | 6.6 |
