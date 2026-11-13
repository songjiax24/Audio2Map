"""Static capability report. No composite chart-quality score is computed."""

from __future__ import annotations

from collections import Counter
from html import escape
import os
from pathlib import Path

from audio2map.eval.v1.reporting import plots
from audio2map.eval.v1.statistics import describe

DISPLAY_SCENARIOS = ("matched_full", "random_full", "none")
TOP_BINS = 15
FINE_AUDIO_VIEW_S = 0.5


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.5g}" if isinstance(value, float) else str(value)


class Page:
    def __init__(self):
        self.parts = []

    def text(self, text):
        self.parts.append(f"<p>{escape(text)}</p>")

    def section(self, title, text):
        self.parts.append(f"<h2>{escape(title)}</h2>")
        self.text(text)

    def table(self, headers, rows):
        self.parts.append("<div class='scroll'><table><thead><tr>" + "".join(
            f"<th>{escape(str(value))}</th>" for value in headers
        ) + "</tr></thead><tbody>")
        for row in rows:
            self.parts.append("<tr>" + "".join(
                f"<td>{escape(fmt(value))}</td>" for value in row
            ) + "</tr>")
        self.parts.append("</tbody></table></div>")

    def figure(self, title, svg):
        self.parts.append(f"<div class='panel'><h3>{escape(title)}</h3>{svg}</div>")


def _matching(page, report):
    page.section("Reference matching", "更高的 F1 是参考复现能力的正向证据；"
                 "同曲同条件允许不同合理编排，F1 不单独决定谱面质量。"
                 "先对同一谱面的种子取均值，再观察跨谱分布；micro 保留事件数加权视角。"
                 "空支持集合沿用 P/R/F1=0 的约定，类别表同时显示参考支持数。")
    scenarios = report["constrained"]
    for unit in ("event", "note"):
        page.parts.append("<div class='cols'>")
        for scope in ("lane_aware", "lane_agnostic"):
            names = [name for name in DISPLAY_SCENARIOS if name in scenarios]
            scores = [scenarios[name]["match"]["views"][unit][scope]["all"] for name in names]
            page.figure(f"{unit} / {scope}", plots.intervals(
                names, [row["chart_distribution"] for row in scores],
            ))
        page.parts.append("</div>")
    page.table(
        ["Scenario", "Unit", "Lane", "Kind", "Micro F1", "Chart mean", "p10", "Median",
         "p90", "Mean CI low", "Mean CI high", "Charts", "Reference support"],
        [
            [name, unit, scope, kind, row["micro"]["f1"], row["chart_distribution"]["mean"],
             row["chart_distribution"]["p10"], row["chart_distribution"]["p50"],
             row["chart_distribution"]["p90"], row["chart_mean_ci"]["low"],
             row["chart_mean_ci"]["high"], row["chart_mean_ci"]["charts"],
             row.get("reference_support", row["micro"]["tp"] + row["micro"]["fn"])]
            for name, scenario in scenarios.items()
            for unit, scopes in scenario["match"]["views"].items()
            for scope, kinds in scopes.items() for kind, row in kinds.items()
        ],
    )


def _teacher(page, report):
    if not report["teacher"]:
        return
    page.section("Teacher forcing", "整首金标前缀；与训练共用 validator。"
                 "合法概率是金标路径上的原始 logits 统计，不等于自由生成合法率。")
    page.table(
        ["Scenario", "Charts", "NLL/token", "NLL/chart", "Accuracy", "PPL", "Legal top1",
         "Legal probability"],
        [[name, row["charts"], row["loss_token"], row["loss_chart"], row["acc_token"],
          row["perplexity"], row["legal_top1_token"], row["legal_prob_token"]]
         for name, row in report["teacher"].items()],
    )
    page.table(
        ["Scenario", "Type", "Tokens", "NLL", "PPL", "Accuracy", "Legal probability"],
        [[name, kind, row["tokens"], row["nll"], row["perplexity"], row["accuracy"],
          row["legal_probability"]] for name, data in report["teacher"].items()
         for kind, row in data["by_type"].items()],
    )
    page.table(
        ["Scenario", "ROW detail", "Correct", "Total", "Rate", "Available charts"],
        [[name, key, value["correct"], value["total"], value["rate"],
          data["row_details"]["available_charts"]]
         for name, data in report["teacher"].items()
         for key, value in data["row_details"]["metrics"].items()],
    )
    if any(not row["row_details"]["available_charts"] for row in report["teacher"].values()):
        page.text("旧 teacher 记录没有逐轨明细；可用 --refresh-teacher 刷新，缺失未当成零。")


def _conditions(page, report):
    page.section("Condition adherence", "每个字段使用原始单位；样本 MAE 和跨谱分布分列。"
                 "任一侧缺失或输入被 mask 的维度不计误差，有效样本数明确列出。")
    page.table(
        ["Scenario", "Field", "Sample MAE", "Observations", "Chart mean", "Median", "p90"],
        [[name, field, value["mae"], value["observations"],
          value["chart_distribution"]["mean"], value["chart_distribution"]["p50"],
          value["chart_distribution"]["p90"]]
         for name, row in report["constrained"].items()
         for field, value in row["condition_error"].items()],
    )
    response = report["condition_response"]
    page.section("Joint-condition response", "固定音频、红线和种子后，替换整组条件。"
                 "排除 donor 自抽与目标未变化的字段；斜率为 ΣΔtarget·Δoutput / ΣΔtarget²。"
                 "整组条件相关，不能据此分离单个字段的因果作用。")
    page.text(f"Seed {response['seed']}; non-self pairs {response['pairs']}; "
              f"identical sequences {response['identical_sequences']}.")
    page.table(
        ["Field", "Pairs", "Missing", "Unchanged target", "Slope", "Same direction", "Rate"],
        [[field, row["pairs"], row["missing"], row["unchanged_target"], row["slope"],
          row["direction_matches"], row["direction_rate"]]
         for field, row in response["fields"].items()],
    )
    page.parts.append("<div class='cols'>")
    for field in ("official_sr", "rework_sr", "msd_overall", "analyzer_ln_percent"):
        row = response["fields"][field]
        page.figure(field, plots.scatter(row["changes"]))
    page.parts.append("</div>")


def _distributions(page, report):
    page.section("Chart distributions", "Human 是同一批目标谱的 v1 token 解码结果。"
                 "频率先汇总计数再除以该项的事件/物件/行总数；不按毫秒重新匹配。"
                 "只展示参考谱最常见的类别时，其余类别合并为 other。")
    data = {"human": report["reference_distribution"]}
    data.update({name: row["distribution"] for name, row in report["constrained"].items()
                 if name in DISPLAY_SCENARIOS})
    for key, title in (("denominator", "Endpoint denominator"), ("position", "Position in beat"),
                       ("hold_grid_span", "Hold span: legal grid points"),
                       ("hold_beat_span", "Hold span: beat-index coordinates"),
                       ("row_pattern", "ROW states")):
        reference = data["human"].get(key) or {}
        labels = [key for key, _ in Counter(reference).most_common(TOP_BINS)]
        all_labels = set().union(*(set(row.get(key) or {}) for row in data.values()))
        other = bool(all_labels - set(labels))
        series = {}
        for name, row in data.items():
            counts = row.get(key) or {}
            total = sum(counts.values())
            if total:
                values = [counts.get(label, 0) / total for label in labels]
                series[name] = values + ([sum(v for k, v in counts.items() if k not in labels)
                                           / total] if other else [])
        if labels:
            page.figure(title, plots.bars(labels + (["other"] if other else []), series))
    labels = [f"lane {lane+1} {kind}" for lane in range(4) for kind in ("tap", "hold")]
    shares, means = {}, {}
    for name, row in data.items():
        counts = [row["lanes"][str(lane)][kind] for lane in range(4) for kind in ("tap", "head")]
        total = sum(counts)
        if total:
            shares[name] = [value / total for value in counts]
        samples = (
            len(report["references"]) if name == "human"
            else report["constrained"][name]["samples"]
        )
        means[name] = [value / samples for value in counts]
    page.figure("Notes by lane: share of all notes", plots.bars(labels, shares))
    page.figure("Notes by lane: mean count per chart/sample", plots.bars(labels, means,
                                                                       xlabel="Mean count"))


def _motifs(page, report):
    if not report["motif"]:
        return
    page.section("Motif frequency", "冻结 train 词表；lane/hand/row 的状态与精确间隔片段。"
                 "长度 1 与更长片段均在词表中。坐标使用 BeatGrid 索引，残拍推进一索引；"
                 "跨红线片段的边界暴露率单独列出。不同 K/坐标的 MAE 不直接横向比较。")
    page.table(["Scenario", "Scope", "K", "MAE", "RMSE", "Spearman"], [
        [name, scope, row["vocabulary"], row["mae"], row["rmse"], row["spearman"]]
        for name, scopes in report["motif"].items() for scope, row in scopes.items()
    ])
    for scope in ("lane", "hand", "row"):
        series = {name: data[scope].get("by_length", {})
                  for name, data in report["motif"].items() if name in DISPLAY_SCENARIOS}
        page.figure(f"{scope}: error contribution and cumulative share",
                    plots.contributions(series))
        full = report["motif"].get("matched_full", {}).get(scope, {})
        gaps = full.get("largest_gaps", [])
        if gaps:
            page.figure(f"{scope}: matched_full largest frequency gaps", plots.bars(
                [row["motif"] for row in gaps],
                {"generated − human": [row["delta"] for row in gaps]},
                xlabel="Frequency difference",
            ))
    page.table(["Corpus", "Scope", "Length", "Windows", "Covered", "Coverage", "Redline crossing"],
               [[name, scope, length, row["windows"], row["matched_windows"], row["coverage"],
                 row["redline_crossing"]] for name, scopes in report["motif_windows"].items()
                for scope, lengths in scopes.items() for length, row in lengths.items()])


def _diversity(page, report):
    if not report["diversity"]:
        return
    page.section("Diversity", "相同音乐与条件允许不同编排。以下为种子两两比较，"
                 "展示变化发生在事件/选轨/完整 hold 的哪些层面；较低或较高的种子间 F1"
                 "都不独立代表质量。每谱十对具有相关性，不能当成十张独立谱。")
    pairs = [pair for row in report["diversity"].values() for pair in row["pairs"]]
    page.text(f"Charts {len(report['diversity'])}; pairs {len(pairs)}; "
              f"identical {sum(pair['identical'] for pair in pairs)}.")
    page.table(["Unit", "Lane", "Pairs", "Mean", "p10", "Median", "p90"], [
        [unit, scope, stats["count"], stats["mean"], stats["p10"], stats["p50"], stats["p90"]]
        for unit in ("event", "note") for scope in ("lane_aware", "lane_agnostic")
        if (stats := describe([pair["match"][unit][scope]["all"]["f1"] for pair in pairs]))
    ])
    page.table(["Motif scope", "Pairs", "Mean MAE", "Median", "p90"], [
        [scope, stats["count"], stats["mean"], stats["p50"], stats["p90"]]
        for scope in ("lane", "hand", "row")
        if (stats := describe([pair.get("motif", {}).get(scope, {}).get("mae") for pair in pairs]))
    ])


def _validity(page, report, rows, out):
    page.section("Structure and validity", "区分整条 token 路径合法、导出物件完整、同轨合理性。"
                 "同轨重叠 gap≤2ms、过近 |gap|<10ms 独立计数，不静默修复。"
                 "次数按样本累加，出现谱面数按源谱去重；参考谱使用相同解码/取整规则。")
    page.text(f"Human reference: {report['reference_plausibility']}")
    page.table(["Scenario", "Samples", "Charts", "Overlap", "Close", "Duplicate start",
                "Nonpositive hold", "Empty", "Affected charts: overlap", "Affected charts: close",
                "Sample overlap rate", "Sample close rate"], [
        [name, row["samples"], row["charts"], row["plausibility"]["concurrent"],
         row["plausibility"]["almost_concurrent"], row["plausibility"]["duplicate_start"],
         row["plausibility"]["nonpositive_hold"], row["plausibility"]["empty_samples"],
         row["plausibility"]["charts_with_concurrent"],
         row["plausibility"]["charts_with_almost_concurrent"],
         row["plausibility"]["sample_rates"]["concurrent"],
         row["plausibility"]["sample_rates"]["almost_concurrent"]]
        for name, row in report["constrained"].items()
    ])
    page.table(["Unconstrained", "Samples", "Valid rate", "Counts", "Illegal median position"],
               [[name, row["samples"], row["valid_rate"], row["counts"], describe([
                   r["position"] for r in rows if r["mode"] == "unconstrained"
                   and r["scenario"] == name and r["validity"] == "illegal_token"
               ])["p50"]] for name, row in report["unconstrained"].items()])
    _failures(page, rows, out)


def _failures(page, rows, out):
    failures = [row for row in rows if row["mode"] == "constrained" and (
        row["notes"] == 0 or row["generated_meta"].get("error")
        or row["plausibility"]["concurrent"] or row["plausibility"]["almost_concurrent"]
    )]
    page.parts.append("<ul>")
    for row in failures:
        target = escape(os.path.relpath(row["output"], out), quote=True)
        label = escape(f"{row['relpath']} / {row['scenario']} / seed {row['seed']}")
        detail = escape(str(row["plausibility"]) + " " + str(row["generated_meta"].get("error")))
        page.parts.append(f"<li><a href='{target}'>{label}</a><br><small>{detail}</small></li>")
    page.parts.append("</ul>")


def write_html(report: dict, rows: list[dict], out: Path) -> None:
    plots.require_plotting()
    page = Page()
    page.parts.append("<h1>Audio2Map v1 capability evaluation</h1>")
    run = report["run"]
    page.text("通过一组测试观察模型能力，不计算自动谱面质量总分。"
              "F1 是参考复现能力的正向证据；条件、分布、多样性与结构指标提供其他视角。")
    page.text(f"Checkpoint: {run['checkpoint']} | split: {run['split']} | "
              f"charts: {len(report['pool']['accepted'])} | subset: {run['subset']} | "
              f"temperature: {run['temperature']} | seeds: {run['seeds']} | "
              f"GPU: {run['gpu'] or run['device']} | bf16/fp32: {run['precision']}")
    page.text("Sampling uses the production generator. No top-p/top-k truncation. "
              "Audio/redlines always belong to the target; rate=1, no mirror.")
    if run.get("rescore"):
        page.text(f"Rescored saved artifacts; source: {run['rescore']['source']}. "
                  f"Analysis elapsed: {report['elapsed_s']:.1f}s; "
                  f"teacher refresh: {run['rescore']['refresh_teacher']}. "
                  "This is not the original generation wall time.")
    page.text(f"Requested/accepted: {len(report['pool']['requested'])}/"
              f"{len(report['pool']['accepted'])}; rejections: {report['pool']['reject_counts']}; "
              f"donor self draws: {report['pool'].get('self_draws', 'see donors.json')}.")
    _teacher(page, report)
    _validity(page, report, rows, out)
    if report["constrained"]:
        _matching(page, report)
        _conditions(page, report)
        _distributions(page, report)
        _motifs(page, report)
        _diversity(page, report)
    page.text("Mean confidence intervals use a fixed-seed 95% chart bootstrap. "
              "p10–p90 describes chart variation, not a confidence interval. "
              "Missing metrics retain their observation counts and never become zero error.")
    _write_page(page, out)


def _write_page(page, out):
    document = """<!doctype html><html lang='zh'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Audio2Map v1 evaluation</title><style>
body{max-width:1120px;margin:auto;padding:40px 24px 80px;font:15px/1.65 system-ui,sans-serif;
color:#25211e;background:#f7f5f2}h1{font-size:30px}h2{margin-top:48px}h3{font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px 10px;
border-bottom:1px solid #ddd7cf;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto;margin:16px 0 28px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{min-width:0}
svg{width:100%;height:auto;background:white}small{color:#665f58}a{color:#244a73}
@media(max-width:800px){.cols{grid-template-columns:1fr}}
</style><body>""" + "\n".join(page.parts) + "</body></html>"
    (out / "index.html").write_text(document, encoding="utf-8")


def write_probe_html(report: dict, out: Path, *, rows=()) -> None:
    plots.require_plotting()
    page = Page()
    page.parts.append("<h1>Audio2Map v1 controlled capability probes</h1>")
    run = report["run"]
    page.text(f"Checkpoint: {run['checkpoint']}; teacher charts {len(run['teacher_charts'])}; "
              f"generation charts {len(run['generation_charts'])}; seeds {run['seeds']}; "
              f"temperature {run['temperature']}; elapsed {report['elapsed_s']:.1f}s.")
    page.text(f"Reused records: {report['reused_records']}; "
              f"copied from compatible prior runs: {report.get('external_reused_records', 0)}. "
              "Elapsed time covers this process, excluding the original cost of reused cases.")
    page.section("Audio alignment", "保持金标、条件、红线、音频长度不变，只循环移动 PCM。"
                 "请求偏移在下表以秒列出，最大为音频长度的 1/4；实际样本偏移记在 records.jsonl。"
                 "该实验测对音频对齐的敏感性，不单独证明音乐编排质量；"
                 "循环拼接点也是扰动的一部分。ΔNLL>0、Δ参考F1<0 支持正确音频对齐有帮助。")
    if report.get("actual_offsets_s"):
        page.table(["Requested shift/s", "Actual minimum/s", "Actual maximum/s"], [
            [offset, row["min"], row["max"]]
            for offset, row in report["actual_offsets_s"].items()
        ])
    fine = [float(value) for value in run["offsets_s"] if abs(value) <= FINE_AUDIO_VIEW_S]
    if len(fine) >= 3:
        page.figure("Small shifts: paired teacher NLL change", plots.offset_curve(
            fine, {condition: [shifts[str(value)]["nll_shift_minus_correct"] for value in fine]
                   for condition, shifts in report["audio_teacher"].items()},
            ylabel="Shifted minus correct NLL / chart",
        ))
        page.figure("Small shifts: paired generation reference F1 change", plots.offset_curve(
            fine, {"matched_full": [report["audio_generation"][str(value)]
                                     ["f1_shift_minus_correct"] for value in fine]},
            ylabel="Shifted minus correct lane-agnostic event F1 / chart",
        ))
    page.table(["Condition", "Requested shift/s", "Charts", "NLL/token", "ROW accuracy",
                "Paired ΔNLL", "95% CI low", "95% CI high"], [
        [condition, offset, row["charts"], row["loss_token"], row["by_type"]["row"]["accuracy"],
         row["nll_shift_minus_correct"]["mean"], row["nll_shift_minus_correct"]["low"],
         row["nll_shift_minus_correct"]["high"]]
        for condition, shifts in report["audio_teacher"].items() for offset, row in shifts.items()
    ])
    page.table(["Requested shift/s", "Charts", "Mean lane-agnostic event F1", "Paired ΔF1",
                "95% CI low", "95% CI high"], [
        [offset, row["charts"], row["f1"]["mean"], row["f1_shift_minus_correct"]["mean"],
         row["f1_shift_minus_correct"]["low"], row["f1_shift_minus_correct"]["high"]]
        for offset, row in report["audio_generation"].items()
    ])
    if any("notes" in row for row in report["audio_generation"].values()):
        page.text("空谱保留在参考 F1 中，组件测量失败按缺失处理。物件数和 token 数单列，"
                  "避免把生成内容的变化全部解释成匹配时机变化；token 数不等于时间覆盖长度。")
        page.table(["Requested shift/s", "Samples", "Empty", "Metadata errors",
                    "Notes min", "Notes median", "Notes max", "Tokens median"], [
            [offset, row["notes"]["count"], row["empty_samples"], row["meta_error_samples"],
             row["notes"]["min"], row["notes"]["p50"], row["notes"]["max"],
             row["tokens"]["p50"]]
            for offset, row in report["audio_generation"].items() if "notes" in row
        ])
    if rows:
        page.table(["Source chart", "Requested shift/s", "Notes", "Tokens", "Warnings"], [
            [row["relpath"], row["offset_s"], row["notes"], row["tokens"], row["warnings"]]
            for row in rows if row.get("kind") == "audio_shift"
        ])
    page.section("Isolated condition requests", "只保留被扫描字段，其余 17 维 mask 为 0，"
                 "避免固定的相关条件与请求值互相矛盾。每条曲线是一张谱面，点为多种子均值。"
                 "单维输入比训练中典型 partial 更稀疏。"
                 "这些少量样本用于观察能力与失败模式，不作总体质量排名。")
    calibration = run.get("protocol", {}).get("definition", {}).get("calibration")
    if calibration:
        page.text(f"Levels calibrated on {len(calibration['selected'])} train charts out of "
                  f"{calibration['available_charts']} accepted charts; "
                  "zero plus q25/q75 of each field's positive observations. "
                  "Pattern shares can overlap and are not normalized across the six classes.")
        page.table(["Pattern", "Measured train charts", "Zero", "Positive", "Request levels"], [
            [field, row["distribution"]["count"], row["zero_count"],
             row["positive_count"], row["levels"]]
            for field, row in calibration["statistics"].items()
        ])
    for field, data in report["control"].items():
        levels = [float(value) for value in data["levels"]]
        page.figure(field, plots.control_curves(levels, data["charts"]))
        page.table(["Requested", "Samples", "Valid metric samples", "Chart mean", "Median",
                    "95% mean CI low", "95% mean CI high"], [
            [value, row["samples"], row["sample_distribution"]["count"],
             row["chart_mean"]["mean"], row["sample_distribution"]["p50"],
             row["chart_mean"]["low"], row["chart_mean"]["high"]]
            for value, row in data["levels"].items()
        ])
        page.table(["Chart", "Complete metrics", "Slope", "Nondecreasing", "High − low"], [
            [path, row["complete"], row["slope"], row["nondecreasing"], row["high_minus_low"]]
            for path, row in data["charts"].items()
        ])
    page.text("All generation uses the production constrained sampler. "
              "F1 is supporting evidence of reference reproduction, not a total quality score. "
              "Confidence intervals resample charts; missing measurements remain missing.")
    if rows:
        page.section("Generated cases requiring inspection", "空谱、组件测量失败和同轨合理性计数"
                     "复用正式报告的判断；下方链接直接打开生成谱面。")
        _failures(page, rows, out)
    _write_page(page, out)
