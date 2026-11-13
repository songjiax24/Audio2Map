"""Optional matplotlib figures embedded as standalone SVG in the evaluation report."""

from __future__ import annotations

from io import StringIO

import numpy as np

from audio2map.utils.time import MS_PER_SECOND

COLORS = ("#244a73", "#c56a2d", "#47846c", "#8263a4", "#7d756b")


def require_plotting() -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("HTML figures require the optional 'eval' extra") from exc


def _figure(height=3.5):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, height), facecolor="white")
    axes = figure.subplots()
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(labelsize=9)
    axes.grid(axis="x", color="#e8e5e0", zorder=0)
    axes.set_axisbelow(True)
    return figure, axes


def _svg(figure):
    from matplotlib import rc_context

    output = StringIO()
    figure.tight_layout()
    with rc_context({"svg.hashsalt": "audio2map-eval-v2", "svg.fonttype": "none"}):
        figure.savefig(output, format="svg", metadata={"Date": None})
    value = output.getvalue()
    return value[value.index("<svg"):]


def bars(labels, series, *, xlabel="Frequency") -> str:
    figure, axes = _figure(max(2.5, len(labels) * 0.34 + 1))
    positions = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    for index, (name, values) in enumerate(series.items()):
        axes.barh(
            positions + (index - (len(series) - 1) / 2) * width,
            values, height=width, label=name, color=COLORS[index % len(COLORS)],
        )
    axes.set_yticks(positions, [label if len(label) < 62 else label[:59] + "…" for label in labels])
    axes.invert_yaxis()
    axes.axvline(0, color="#aaa49b", linewidth=0.6)
    axes.set_xlabel(xlabel)
    axes.legend(fontsize=8, loc="best", frameon=False)
    return _svg(figure)


def intervals(labels, stats) -> str:
    figure, axes = _figure(max(2.5, len(labels) * 0.55 + 1))
    for index, row in enumerate(stats):
        if row["mean"] is None:
            continue
        color = COLORS[index % len(COLORS)]
        axes.barh(index, row["mean"], height=0.5, color=color, alpha=0.4)
        axes.plot([row["p10"], row["p90"]], [index, index], color=color, linewidth=2)
        axes.scatter([row["p50"]], [index], color=color, s=22, zorder=3)
    axes.set_yticks(range(len(labels)), labels)
    axes.invert_yaxis()
    axes.set_xlim(0, 1)
    axes.set_xlabel("F1: bar = chart mean; line = p10–p90; dot = median")
    return _svg(figure)


def scatter(changes) -> str:
    figure, axes = _figure()
    x = [row["target_delta"] for row in changes]
    y = [row["generated_delta"] for row in changes]
    axes.scatter(x, y, alpha=0.75, s=20, color=COLORS[0])
    extent = max([abs(value) for value in x + y], default=1) or 1
    axes.plot([-extent, extent], [-extent, extent], "--", color=COLORS[1], label="y = x")
    axes.axvline(0, color="#aaa49b", linewidth=0.6)
    axes.axhline(0, color="#aaa49b", linewidth=0.6)
    axes.set_xlabel("Requested condition change")
    axes.set_ylabel("Measured output change")
    axes.legend(fontsize=8, frameon=False)
    return _svg(figure)


def contributions(series) -> str:
    lengths = sorted({int(key) for rows in series.values() for key in rows})
    figure, axes = _figure()
    cumulative_axes = axes.twinx()
    width = 0.8 / max(1, len(series))
    positions = np.arange(len(lengths))
    for index, (name, rows) in enumerate(series.items()):
        values = np.asarray([rows.get(str(length), {}).get("mae_contribution", 0)
                             for length in lengths])
        color = COLORS[index % len(COLORS)]
        axes.bar(positions + (index - (len(series) - 1) / 2) * width,
                 values, width=width, color=color, label=name, alpha=0.7)
        if values.sum():
            cumulative_axes.plot(positions, np.cumsum(values) / values.sum(), color=color)
    axes.set_xticks(positions, [str(length) for length in lengths])
    axes.set_xlabel("Motif length")
    axes.set_ylabel("Contribution to total MAE")
    cumulative_axes.set_ylim(0, 1.05)
    cumulative_axes.set_ylabel("Cumulative share")
    axes.legend(fontsize=8, frameon=False)
    return _svg(figure)


def control_curves(levels, responses) -> str:
    figure, axes = _figure()
    for row in responses.values():
        if row["complete"]:
            axes.plot(levels, [row["values"][str(value)] for value in levels],
                      color=COLORS[0], alpha=0.35, marker="o", markersize=3)
    axes.plot(levels, levels, "--", color=COLORS[1], label="requested = measured")
    axes.set_xlabel("Requested value (other condition fields masked)")
    axes.set_ylabel("Measured output: mean over seeds")
    axes.legend(fontsize=8, frameon=False)
    return _svg(figure)


def offset_curve(offsets_s, series, *, ylabel) -> str:
    figure, axes = _figure()
    x = [value * MS_PER_SECOND for value in offsets_s]
    for index, (name, rows) in enumerate(series.items()):
        means = [np.nan if row["mean"] is None else row["mean"] for row in rows]
        low = [np.nan if row["low"] is None else row["low"] for row in rows]
        high = [np.nan if row["high"] is None else row["high"] for row in rows]
        color = COLORS[index % len(COLORS)]
        axes.plot(x, means, marker="o", markersize=3, color=color, label=name)
        axes.fill_between(x, low, high, alpha=0.15, color=color)
    axes.axhline(0, color="#aaa49b", linewidth=0.6)
    axes.axvline(0, color="#aaa49b", linewidth=0.6)
    axes.set_xlabel("Requested circular audio shift / ms")
    axes.set_ylabel(ylabel)
    axes.legend(fontsize=8, frameon=False)
    return _svg(figure)
