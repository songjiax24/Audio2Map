#!/usr/bin/env python3
"""Build a reproducible corpus report for one finished v0 baseline run.

Human reference statistics are recomputed from the target charts named in the
matched_full result, with the same accumulators the evaluator uses. Stored
motif distances are checked against that recomputation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

from audio2map.eval.chart_stats import ChartStatAccumulator
from audio2map.eval.motif import motif_length, parse_motif
from audio2map.eval.motif_freq import (
    MotifCorpus,
    frequency_mae,
    frequency_rmse,
    load_ordered_motif_vocab,
    spearman,
)
from audio2map.grid import CanonicalTiming
from audio2map.osu.parser import parse_beatmap

MATCH_TREE = {
    "event": ("tap", "hold_head", "hold_tail", "all_event"),
    "note": ("tap", "hold", "all_notes"),
}
SIDES = ("lane_aware", "lane_agnostic")
SIDE_LABEL = {"lane_aware": "lane-aware", "lane_agnostic": "lane-agnostic"}
ADHERENCE_FIELDS = (
    "official_sr",
    "analyzer_ln_percent",
    "analyzer_hb_row_ratio",
    "analyzer_stream",
    "analyzer_chordstream",
    "analyzer_jacks",
    "analyzer_coordination",
    "analyzer_density",
    "analyzer_wildcard",
    "msd_overall",
    "msd_stream",
    "msd_jumpstream",
    "msd_handstream",
    "msd_stamina",
    "msd_jack_speed",
    "msd_chordjack",
    "msd_technical",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if abs(value) != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def constrained_row(report: dict, scenario: str) -> dict:
    rows = report["scenarios"][scenario]["constrained"]
    if len(rows) != 1 or rows[0]["seed"] != 0:
        raise SystemExit(f"{scenario} constrained result is not a single seed-0 block")
    return rows[0]


def log(message: str) -> None:
    print(message, flush=True)


def human_corpus(paths: list[str], vocab) -> tuple[dict, MotifCorpus]:
    stats = ChartStatAccumulator()
    motifs = MotifCorpus(vocab)
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        beatmap = parse_beatmap(Path(path))
        timing = CanonicalTiming.from_beatmap(beatmap)
        stats.add_chart(beatmap.notes, timing)
        motifs.add_chart(beatmap.notes, timing)
        if index % 100 == 0 or index == total:
            log(f"human charts {index}/{total}")
    return stats.to_dict(), motifs


def check_motif_distance(stored: dict, generated: dict, human: MotifCorpus) -> list[dict]:
    checks = []
    for scope in ("lane", "hand", "row"):
        gen = generated["scopes"][scope]["frequency"]
        ref = human.frequency(scope)
        texts = human.vocab.texts[scope]
        recomputed = {
            "mae": frequency_mae(gen, ref),
            "rmse": frequency_rmse(gen, ref),
            "spearman": spearman(gen, ref),
        }
        for name, value in recomputed.items():
            saved = stored[scope][name]
            if saved is None or value is None or abs(saved - value) > 1e-8:
                raise SystemExit(
                    f"motif {scope} {name} recomputed {value} != stored {saved}"
                )
        checks.append({"scope": scope, **{f"stored_{k}": stored[scope][k] for k in recomputed}})
    return checks


def match_summary(charts: list[dict]) -> list[dict]:
    rows = []
    for level, keys in MATCH_TREE.items():
        for side in SIDES:
            for key in keys:
                block = [chart["match"][level][side][key] for chart in charts]
                for metric in ("precision", "recall", "f1"):
                    values = [item[metric] for item in block]
                    rows.append(
                        {
                            "level": level,
                            "side": side,
                            "key": key,
                            "metric": metric,
                            "mean": mean(values),
                            "p10": percentile(values, 0.10),
                            "p50": percentile(values, 0.50),
                            "p90": percentile(values, 0.90),
                        }
                    )
    return rows


def adherence_summary(charts: list[dict]) -> list[dict]:
    rows = []
    for field in ADHERENCE_FIELDS:
        values = [chart["adherence"][field] for chart in charts]
        present = [value for value in values if value is not None]
        rows.append(
            {
                "field": field,
                "n": len(present),
                "missing": len(values) - len(present),
                "mean": mean(present),
                "p10": percentile(present, 0.10),
                "p50": percentile(present, 0.50),
                "p90": percentile(present, 0.90),
            }
        )
    return rows


def hold_tail_from_charts(charts: list[dict]) -> dict:
    instances = 0
    charts_with = 0
    for chart in charts:
        count = sum(1 for issue in chart["decode_issues"] if issue.get("kind") == "missing_hold_tail")
        instances += count
        charts_with += int(count > 0)
    return {"charts_with": charts_with, "instances": instances, "charts": len(charts)}


def length_counts(frequency: list[float], observations: int, texts: tuple[str, ...]) -> dict[int, float]:
    counts: dict[int, float] = {}
    for motif, freq in zip(texts, frequency):
        length = motif_length(parse_motif(motif))
        counts[length] = counts.get(length, 0.0) + freq * observations
    return counts


def mae_by_length(generated: list[float], human: list[float], texts: tuple[str, ...]) -> list[dict]:
    """Each length's additive share of frequency MAE.

    MAE is the mean of ``|f_gen - f_human|`` over the vocabulary. A length's
    contribution is that sum restricted to motifs of the length, still divided
    by the vocabulary size, so the contributions add up to the MAE.
    """
    if not (len(generated) == len(human) == len(texts)) or not texts:
        raise ValueError("frequency vectors must stay aligned with the vocabulary")
    absolute: dict[int, float] = {}
    counts: dict[int, int] = {}
    for motif, f_gen, f_human in zip(texts, generated, human):
        length = motif_length(parse_motif(motif))
        absolute[length] = absolute.get(length, 0.0) + abs(f_gen - f_human)
        counts[length] = counts.get(length, 0) + 1
    vocab = len(texts)
    total = sum(absolute.values())
    rows = []
    for length in sorted(absolute):
        piece = absolute[length]
        rows.append(
            {
                "length": length,
                "motifs": counts[length],
                "contribution": piece / vocab,
                "share": piece / total if total else 0.0,
            }
        )
    return rows


PALETTE = ("#244a73", "#c56a2d", "#2f6d4f", "#6b5b95")
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
KEY_LABEL = {
    "tap": "tap",
    "hold": "hold",
    "hold_head": "hold head",
    "hold_tail": "hold tail",
    "all_event": "all",
    "all_notes": "all",
}
FIELD_LABEL = {
    "official_sr": "official SR",
    "analyzer_ln_percent": "LN %",
    "analyzer_hb_row_ratio": "HB row",
    "analyzer_stream": "stream",
    "analyzer_chordstream": "chordstream",
    "analyzer_jacks": "jacks",
    "analyzer_coordination": "coordination",
    "analyzer_density": "density",
    "analyzer_wildcard": "wildcard",
    "msd_overall": "overall",
    "msd_stream": "stream",
    "msd_jumpstream": "jumpstream",
    "msd_handstream": "handstream",
    "msd_stamina": "stamina",
    "msd_jack_speed": "jack speed",
    "msd_chordjack": "chordjack",
    "msd_technical": "technical",
}


def axis_scale(peak: float) -> tuple[float, str]:
    if 0 < peak < 0.01:
        exp = math.floor(math.log10(peak) + 1e-12)
        return 10**exp, " ×10" + str(exp).translate(_SUP)
    return 1.0, ""


def axis_text(value: float, scale: float) -> str:
    shown = 0.0 if abs(value) < 1e-15 else value / scale
    if abs(shown) < 1e-9:
        return "0"
    if abs(shown - round(shown)) < 1e-8:
        return str(int(round(shown)))
    text = f"{shown:.4f}".rstrip("0").rstrip(".")
    return text


def nice_limit(peak: float) -> float:
    """Round an upper bound up to a 1-2-5 step so ticks stay readable."""
    if peak <= 0:
        return 1.0
    raw = peak / 4
    exp = math.floor(math.log10(raw) + 1e-12)
    base = 10**exp
    frac = raw / base
    step = next(n * base for n in (1, 2, 2.5, 5, 10) if n >= frac - 1e-9)
    limit = math.ceil(peak / step - 1e-9) * step
    if limit <= peak * 1.02:
        limit += step
    return limit


def axis_ticks(limit: float) -> list[float]:
    if limit <= 0:
        limit = 1.0
    raw = limit / 4
    exp = math.floor(math.log10(raw) + 1e-12)
    base = 10**exp
    best: tuple[tuple[int, int], float, int] | None = None
    for shift in (0, -1):
        unit = base * 10**shift
        for n in (1, 2, 2.5, 5, 10):
            candidate = n * unit
            count = limit / candidate
            nearest = round(count)
            if not 3 <= nearest <= 5 or abs(count - nearest) > 1e-6:
                continue
            score = (abs(nearest - 4), nearest)
            if best is None or score < best[0]:
                best = (score, candidate, nearest)
    step, intervals = (limit / 4, 4) if best is None else (best[1], best[2])
    return [step * i for i in range(intervals + 1)]


def span_ticks(lo: float, hi: float) -> list[float]:
    if lo >= 0:
        return axis_ticks(hi)
    if hi <= 0:
        return [-tick for tick in reversed(axis_ticks(-lo))]
    step = axis_ticks(max(hi, -lo))
    step = step[1] - step[0]
    start = math.ceil(lo / step - 1e-9)
    end = math.floor(hi / step + 1e-9)
    return [step * i for i in range(start, end + 1)]


def chosen_limit(peak: float, forced: float | None) -> float:
    if forced is not None and peak <= forced + 1e-9:
        return forced
    return nice_limit(peak if forced is None else max(peak, forced))


def legend_html(names: list[str]) -> str:
    parts = [
        f'<span class="swatch"><i style="background:{PALETTE[index % len(PALETTE)]}"></i>{esc(name)}</span>'
        for index, name in enumerate(names)
    ]
    return f'<p class="legend">{"".join(parts)}</p>'


def frame(svg: str, names: list[str]) -> str:
    return f'<div class="frame">{svg}{legend_html(names)}</div>'


def grid(*items: str) -> str:
    return '<div class="cols">' + "".join(items) + "</div>"


def panel(title: str, body: str) -> str:
    return f'<div class="panel"><h3>{title}</h3>{body}</div>'


def svg_grouped(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    y_label: str,
    width: int = 760,
    height: int = 280,
    rotate: bool | None = None,
    y_max: float | None = None,
) -> str:
    if not categories or not series:
        return ""
    if rotate is None:
        rotate = len(categories) > 12 or max(len(category) for category in categories) > 8
    left, right, top = 52, 12, 22
    bottom = 78 if rotate else 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max((max(values) if values else 0) for _, values in series)
    limit = chosen_limit(peak, y_max)
    scale, suffix = axis_scale(limit)
    ticks = axis_ticks(limit)
    group = plot_w / len(categories)
    bar = group / (len(series) + 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
        f'<text x="{left}" y="14" class="axis">{esc(y_label + suffix)}</text>',
    ]
    for value in ticks:
        y = top + plot_h - (plot_h * value / limit)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis">{axis_text(value, scale)}</text>')
    for index, category in enumerate(categories):
        for series_index, (_, values) in enumerate(series):
            value = values[index]
            bar_h = 0 if limit == 0 else plot_h * value / limit
            x = left + index * group + bar * (series_index + 0.5)
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar * 0.72:.1f}" height="{max(bar_h, 0):.1f}" '
                f'fill="{PALETTE[series_index % len(PALETTE)]}"/>'
            )
        anchor_x = left + index * group + group / 2
        if rotate:
            parts.append(
                f'<text text-anchor="end" class="axis" transform="translate({anchor_x:.1f},{height - bottom + 14}) rotate(-55)">{esc(category)}</text>'
            )
        else:
            parts.append(
                f'<text x="{anchor_x:.1f}" y="{height - 10}" text-anchor="middle" class="axis">{esc(category)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def svg_hist(
    values: list[float],
    *,
    title: str,
    bins: int = 20,
    width: int = 520,
    height: int = 220,
    x_min: float | None = None,
    x_max: float | None = None,
) -> str:
    if not values:
        return ""
    lo = min(values) if x_min is None else min(x_min, min(values))
    hi = max(values) if x_max is None else max(x_max, max(values))
    if x_min is not None and min(values) >= x_min:
        lo = x_min
    if x_max is not None and max(values) <= x_max + 1e-9:
        hi = x_max
    if hi == lo:
        hi = lo + 1
    width_bin = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lo) / width_bin))
        counts[index] += 1
    left, right, top, bottom = 44, 12, 22, 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    limit = nice_limit(max(counts))
    bar = plot_w / bins
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
        f'<text x="{left}" y="14" class="axis">{esc(title)}</text>',
    ]
    for index, count in enumerate(counts):
        bar_h = plot_h * count / limit
        x = left + index * bar
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar * 0.86:.1f}" height="{bar_h:.1f}" fill="{PALETTE[0]}"/>')
    for value in axis_ticks(limit):
        y = top + plot_h - plot_h * value / limit
        parts.append(f'<text x="{left - 4}" y="{y + 3:.1f}" text-anchor="end" class="axis">{axis_text(value, 1)}</text>')
    for value in axis_ticks(hi - lo):
        x = left + plot_w * (value / (hi - lo))
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" class="axis">{axis_text(lo + value, 1)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_range(
    categories: list[str],
    series: list[tuple[str, list[float], list[float], list[float]]],
    *,
    y_label: str,
    width: int = 980,
    height: int = 320,
    y_max: float | None = None,
) -> str:
    """Vertical p10–p90 stroke with a p50 dot for each series."""
    if not categories or not series:
        return ""
    rotate = len(categories) > 6 or max(len(category) for category in categories) > 12
    left, right, top = 52, 12, 22
    bottom = 78 if rotate else 28
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max((max(highs) if highs else 0) for _, _, _, highs in series)
    limit = chosen_limit(peak, y_max)
    scale, suffix = axis_scale(limit)
    group = plot_w / len(categories)
    slot = group / (len(series) + 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
        f'<text x="{left}" y="14" class="axis">{esc(y_label + suffix)}</text>',
    ]
    for value in axis_ticks(limit):
        y = top + plot_h - plot_h * value / limit
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis">{axis_text(value, scale)}</text>')
    for index, category in enumerate(categories):
        for series_index, (_, lows, mids, highs) in enumerate(series):
            color = PALETTE[series_index % len(PALETTE)]
            cx = left + index * group + slot * (series_index + 1)
            y_low = top + plot_h - plot_h * lows[index] / limit
            y_high = top + plot_h - plot_h * highs[index] / limit
            y_mid = top + plot_h - plot_h * mids[index] / limit
            parts.append(f'<line x1="{cx:.1f}" y1="{y_high:.1f}" x2="{cx:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<circle cx="{cx:.1f}" cy="{y_mid:.1f}" r="3.5" fill="{color}"/>')
        anchor_x = left + index * group + group / 2
        if rotate:
            parts.append(
                f'<text text-anchor="end" class="axis" transform="translate({anchor_x:.1f},{height - bottom + 14}) rotate(-55)">{esc(category)}</text>'
            )
        else:
            parts.append(f'<text x="{anchor_x:.1f}" y="{height - 10}" text-anchor="middle" class="axis">{esc(category)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_lines(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    y_label: str,
    width: int = 980,
    height: int = 280,
    label_stride: int = 1,
    y_max: float | None = None,
) -> str:
    if not categories or not series:
        return ""
    left, right, top, bottom = 52, 12, 22, 28
    plot_w = width - left - right
    plot_h = height - top - bottom
    peak = max((max(values) if values else 0) for _, values in series)
    limit = chosen_limit(peak, y_max)
    scale, suffix = axis_scale(limit)
    step = plot_w / max(len(categories) - 1, 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
        f'<text x="{left}" y="14" class="axis">{esc(y_label + suffix)}</text>',
    ]
    for value in axis_ticks(limit):
        y = top + plot_h - plot_h * value / limit
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 3:.1f}" text-anchor="end" class="axis">{axis_text(value, scale)}</text>')
    for series_index, (_, values) in enumerate(series):
        color = PALETTE[series_index % len(PALETTE)]
        points = []
        for index, value in enumerate(values):
            x = left + index * step
            y = top + plot_h - plot_h * value / limit
            points.append(f"{x:.1f},{y:.1f}")
        dash = ("", "6 3", "1.5 2.5")[series_index % 3]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.75" '
            f'stroke-dasharray="{dash}" points="{" ".join(points)}"/>'
        )
    for index, category in enumerate(categories):
        if index % label_stride:
            continue
        x = left + index * step
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" text-anchor="middle" class="axis">{esc(category)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# Advance widths of 11px DejaVu Sans, the report's axis font.
_AXIS_ADVANCE = {
    " ": 3.06, "+": 6.43, "-": 3.67, "/": 4.00, "_": 7.20, "·": 3.67,
    "0": 6.12, "1": 6.12, "2": 6.12, "3": 6.12, "4": 6.12,
    "5": 6.12, "6": 6.12, "7": 6.12, "8": 6.12, "9": 6.12,
    "A": 8.00, "B": 7.34, "C": 8.00, "D": 7.95, "E": 7.34, "F": 6.80, "G": 8.56,
    "H": 7.95, "I": 3.06, "J": 5.50, "K": 7.60, "L": 6.12, "M": 9.17, "N": 7.95,
    "O": 8.56, "P": 7.34, "Q": 8.80, "R": 8.40, "S": 7.34, "T": 6.80, "U": 7.95,
    "V": 8.00, "W": 10.80, "X": 8.00, "Y": 8.00, "Z": 6.80,
    "a": 6.12, "b": 6.12, "c": 5.60, "d": 6.12, "e": 6.12, "f": 4.00, "g": 6.12,
    "h": 6.12, "i": 2.45, "j": 3.25, "k": 6.00, "l": 2.45, "m": 9.17, "n": 6.12,
    "o": 6.12, "p": 6.12, "q": 6.12, "r": 4.00, "s": 5.60, "t": 3.20, "u": 6.12,
    "v": 5.60, "w": 8.80, "x": 6.40, "y": 6.00, "z": 5.60,
}


def category_margin(categories: list[str]) -> float:
    """Left inset so 11px labels end just before the plot."""
    def width(text: str) -> float:
        return sum(_AXIS_ADVANCE.get(char, 7.0) for char in text)

    return 18 + max(width(category) for category in categories)


def svg_horizontal(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    *,
    x_label: str,
    width: int = 980,
    diverging: bool = False,
    x_max: float | None = None,
) -> str:
    if not categories or not series:
        return ""
    right, top, bottom = 18, 22, 28
    row_h = 14 * len(series) + 10
    height = top + row_h * len(categories) + bottom
    data_lo = min(min(values) if values else 0 for _, values in series)
    data_hi = max(max(values) if values else 0 for _, values in series)
    if diverging:
        lo = -nice_limit(-data_lo) if data_lo < 0 else 0.0
        hi = nice_limit(data_hi) if data_hi > 0 else 0.0
        if x_max is not None:
            lo = max(lo, -x_max)
            hi = min(hi, x_max)
    else:
        lo = 0.0
        hi = chosen_limit(max(data_hi, 0.0), x_max)
    left = category_margin(categories) + (14 if lo < 0 else 0)
    plot_w = width - left - right
    span = hi - lo or 1
    scale, suffix = axis_scale(max(abs(lo), abs(hi), 1e-15))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart">',
        f'<text x="{left}" y="14" class="axis">{esc(x_label + suffix)}</text>',
    ]
    axis_y = height - bottom
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" class="axis-line"/>')
    for value in span_ticks(lo, hi):
        x = left + plot_w * (value - lo) / span
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{axis_y}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{axis_y + 14}" text-anchor="middle" class="axis">{axis_text(value, scale)}</text>')
    if lo < 0 < hi:
        zero_x = left + plot_w * (0 - lo) / span
        parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{axis_y}" class="axis-line"/>')
    for index, category in enumerate(categories):
        y0 = top + index * row_h
        parts.append(f'<text x="{left - 8}" y="{y0 + row_h / 2 + 3:.1f}" text-anchor="end" class="axis">{esc(category)}</text>')
        bar_h = 8
        gap = (row_h - 8 - bar_h * len(series)) / 2
        for series_index, (_, values) in enumerate(series):
            value = values[index]
            y = y0 + gap + series_index * (bar_h + 3)
            x0 = left + plot_w * (min(value, 0) - lo) / span
            x1 = left + plot_w * (max(value, 0) - lo) / span
            parts.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1 - x0, 0.8):.1f}" height="{bar_h}" '
                f'fill="{PALETTE[series_index % len(PALETTE)]}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def mae_length_section(by_scope: dict[str, list[dict]]) -> str:
    """Charts of each length's additive share of motif MAE."""
    parts = [
        "<h3>Where the error comes from</h3>",
        "<p class=\"lead\">A length's contribution is the sum of absolute frequency differences for motifs of that length, divided by the vocabulary size. Within a scope these contributions add up to the MAE. Bars run through the longest length that still accounts for at least 0.5% of the MAE. The line is the cumulative share.</p>",
    ]
    prepared = []
    for scope, rows in by_scope.items():
        visible = [row for row in rows if row["matched_share"] >= 0.005 or row["random_share"] >= 0.005]
        last = max(row["length"] for row in visible) if visible else rows[-1]["length"]
        prepared.append((scope, [row for row in rows if row["length"] <= last]))
    contribution_max = nice_limit(
        max(row[key] for _, shown in prepared for row in shown for key in ("matched_contribution", "random_contribution"))
    )
    for scope, shown in prepared:
        labels = [str(row["length"]) for row in shown]
        matched_cum = []
        random_cum = []
        matched_total = 0.0
        random_total = 0.0
        for row in shown:
            matched_total += row["matched_share"]
            random_total += row["random_share"]
            matched_cum.append(matched_total * 100)
            random_cum.append(random_total * 100)
        series = [
            ("matched", [row["matched_contribution"] for row in shown]),
            ("random", [row["random_contribution"] for row in shown]),
        ]
        parts.append(f"<h3>{esc(scope)}</h3>")
        parts.append(
            grid(
                frame(
                    svg_grouped(labels, series, y_label="Contribution to MAE", width=520, height=280, y_max=contribution_max),
                    ["matched", "random"],
                ),
                frame(
                    svg_lines(
                        labels,
                        [("matched", matched_cum), ("random", random_cum)],
                        y_label="Cumulative share %",
                        width=520,
                        height=280,
                        label_stride=max(1, len(labels) // 8),
                        y_max=100,
                    ),
                    ["matched", "random"],
                ),
            )
        )
    return "".join(parts)


def paired_tables(
    title_left: str,
    headers_left: list[str],
    rows_left: list[list[object]],
    title_right: str,
    headers_right: list[str],
    rows_right: list[list[object]],
) -> str:
    return (
        f"<h3>{esc(title_left)}</h3>{table(headers_left, rows_left)}"
        f"<h3>{esc(title_right)}</h3>{table(headers_right, rows_right)}"
    )


def section(index: str, title: str, body: str, anchor: str) -> str:
    return f'<section id="{esc(anchor)}"><h2><span class="idx">{esc(index)}</span>{esc(title)}</h2>{body}</section>'


def build_html(context: dict) -> str:
    nav = [
        ("validity", "Validity"),
        ("teacher", "Teacher"),
        ("match", "Matching"),
        ("adherence", "Condition"),
        ("distribution", "Distribution"),
        ("motif", "Motif"),
        ("diversity", "Diversity"),
    ]
    links = "".join(f'<a href="#{anchor}">{esc(label)}</a>' for anchor, label in nav)
    pages = [
        ("01", "Are the charts legal", context["validity_html"], "validity"),
        ("02", "Teacher forcing", context["teacher_html"], "teacher"),
        ("03", "Matching the human chart", context["match_html"], "match"),
        ("04", "Condition error", context["adherence_html"], "adherence"),
        ("05", "Chart distribution", context["distribution_html"], "distribution"),
        ("06", "Motif frequency", context["motif_html"], "motif"),
        ("07", "Diversity across samples", context["diversity_html"], "diversity"),
    ]
    style = """
:root { --ink:#1c1916; --muted:#6f6962; --rule:#e3ddd4; --paper:#f7f5f2; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:1040px; padding:56px 32px 96px; color:var(--ink); background:var(--paper);
  font:15.5px/1.6 "DejaVu Sans",sans-serif; }
h1 { font-size:34px; font-weight:600; letter-spacing:-0.03em; line-height:1.15; margin:0 0 12px; }
.lede { margin:0; color:#3e3a36; }
nav { display:flex; flex-wrap:wrap; gap:6px; margin:28px 0 0; }
nav a { color:var(--ink); text-decoration:none; font-size:13px; padding:5px 11px; border-radius:999px; background:#efece6; }
nav a:hover { background:#e4dfd6; }
section { margin-top:64px; }
h2 { font-size:22px; font-weight:600; letter-spacing:-0.02em; margin:0 0 8px; }
h2 .idx { color:var(--muted); font-weight:500; margin-right:10px; font-variant-numeric:tabular-nums; }
.lead { margin:0 0 16px; color:#3e3a36; }
h3 { font-size:14px; font-weight:600; margin:20px 0 6px; color:#2a2724; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:6px 32px; align-items:start; }
.panel h3 { margin-top:0; }
.frame { margin:0 0 4px; }
svg.chart { width:100%; height:auto; display:block; }
.axis { font-size:11px; fill:#6f6962; font-family:"DejaVu Sans",sans-serif; }
.gridline, .grid { stroke:#ece7df; }
.axis-line { stroke:#d5cfc6; }
.legend { margin:0 0 14px; color:var(--muted); font-size:12px; }
.swatch { display:inline-flex; align-items:center; margin-right:14px; }
.swatch i { width:9px; height:9px; border-radius:2px; display:inline-block; margin-right:6px; }
table { width:100%; border-collapse:collapse; margin:4px 0 20px; font-size:14px; font-variant-numeric:tabular-nums; }
th { text-align:right; font-weight:600; font-size:12px; color:var(--muted); border-bottom:1px solid #cfc8be; padding:8px 10px 6px; }
td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule); }
th:first-child, td:first-child { text-align:left; }
@media (max-width:820px) {
  body { padding:28px 16px 64px; }
  .cols { grid-template-columns:1fr; }
}
"""
    body = [section(index, title, html_body, anchor) for index, title, html_body, anchor in pages]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        "<title>v0 baseline</title><style>"
        + style
        + "</style></head><body><h1>v0 baseline</h1>"
        + f"<p class=\"lede\">{context['lede']}</p><nav>{links}</nav>"
        + "".join(body)
        + "</body></html>"
    )


def _triple(human: dict, matched: dict, random_stats: dict, section: str, lane: str, state: str, name: str, metric: str) -> list[float]:
    return [
        human[section][metric][lane][state].get(name, 0),
        matched[section][metric][lane][state].get(name, 0),
        random_stats[section][metric][lane][state].get(name, 0),
    ]


def _compare_bars(
    categories: list[str],
    columns: list[list[float]],
    y_label: str,
    *,
    width: int = 500,
    height: int = 250,
    y_max: float | None = None,
) -> str:
    series = [
        ("human", [column[0] for column in columns]),
        ("matched", [column[1] for column in columns]),
        ("random", [column[2] for column in columns]),
    ]
    return frame(
        svg_grouped(categories, series, y_label=y_label, width=width, height=height, rotate=False, y_max=y_max),
        [name for name, _ in series],
    )


def snap_charts(human: dict, matched: dict, random_stats: dict) -> str:
    order = ("1/1", "1/2", "1/3", "1/4", "1/6", "1/8", "1/12", "1/16", "1/24", "1/48")
    present = human["snap"]["pooled_frequency"]["all_lanes"]["all_event"]
    names = [name for name in order if name in present]
    labels = {"all_event": "All events", "tap": "Tap", "hold_head": "Hold head", "hold_tail": "Hold tail"}
    columns_by_state = {
        state: [_triple(human, matched, random_stats, "snap", "all_lanes", state, name, "pooled_frequency") for name in names]
        for state in ("all_event", "tap", "hold_head", "hold_tail")
    }
    def ceiling(*states: str) -> float:
        return nice_limit(max(value for state in states for column in columns_by_state[state] for value in column))

    dense = ceiling("all_event", "tap")
    sparse = ceiling("hold_head", "hold_tail")
    limits = {"all_event": dense, "tap": dense, "hold_head": sparse, "hold_tail": sparse}
    panels = [
        panel(labels[state], _compare_bars(names, columns, "Share", y_max=limits[state]))
        for state, columns in columns_by_state.items()
    ]
    return grid(*panels)


def note_charts(human: dict, matched: dict, random_stats: dict) -> str:
    lanes = ["lane_1", "lane_2", "lane_3", "lane_4"]
    labels = ["lane 1", "lane 2", "lane 3", "lane 4"]
    averages_by_kind = {}
    shares_by_kind = {}
    for kind in ("tap", "hold"):
        averages_by_kind[kind] = [
            [human["note"]["avg_count"][lane][kind], matched["note"]["avg_count"][lane][kind], random_stats["note"]["avg_count"][lane][kind]]
            for lane in lanes
        ]
        shares_by_kind[kind] = [
            [human["note"]["pooled_frequency"][lane][kind], matched["note"]["pooled_frequency"][lane][kind], random_stats["note"]["pooled_frequency"][lane][kind]]
            for lane in lanes
        ]
    count_max = nice_limit(max(value for rows in averages_by_kind.values() for row in rows for value in row))
    share_max = nice_limit(max(value for rows in shares_by_kind.values() for row in rows for value in row))
    panels = []
    for kind, title in (("tap", "Tap"), ("hold", "Hold")):
        panels.append(panel(f"{title} mean count", _compare_bars(labels, averages_by_kind[kind], "Per chart", y_max=count_max)))
        panels.append(panel(f"{title} share of all notes", _compare_bars(labels, shares_by_kind[kind], "Share", y_max=share_max)))
    return grid(*panels)


def tick_charts(human: dict, matched: dict, random_stats: dict, tick_names: list[str] | None = None) -> str:
    if tick_names is None:
        tick_names = sorted(human["tick"]["pooled_frequency"]["all_lanes"]["all_event"], key=int)
    labels = {"all_event": "All events", "tap": "Tap", "hold_head": "Hold head", "hold_tail": "Hold tail"}
    series_by_state = {}
    for state in ("all_event", "tap", "hold_head", "hold_tail"):
        columns = [_triple(human, matched, random_stats, "tick", "all_lanes", state, name, "pooled_frequency") for name in tick_names]
        series_by_state[state] = [
            ("human", [column[0] for column in columns]),
            ("matched", [column[1] for column in columns]),
            ("random", [column[2] for column in columns]),
        ]
    def ceiling(*states: str) -> float:
        return nice_limit(
            max(value for state in states for _, values in series_by_state[state] for value in values)
        )

    dense = ceiling("all_event", "tap")
    sparse = ceiling("hold_head", "hold_tail")
    limits = {"all_event": dense, "tap": dense, "hold_head": sparse, "hold_tail": sparse}
    panels = [
        panel(
            labels[state],
            frame(
                svg_lines(tick_names, series, y_label="Share", width=500, height=240, label_stride=4, y_max=limits[state]),
                [name for name, _ in series],
            ),
        )
        for state, series in series_by_state.items()
    ]
    return grid(*panels)


def adherence_charts(matched_rows: list[dict], random_rows: list[dict]) -> str:
    def chart(fields: tuple[str, ...], value_key: str, x_max: float) -> str:
        labels = [FIELD_LABEL.get(field, field) for field in fields]
        left = {row["field"]: row[value_key] for row in matched_rows}
        right = {row["field"]: row[value_key] for row in random_rows}
        return frame(
            svg_horizontal(
                labels,
                [("matched", [left[field] for field in fields]), ("random", [right[field] for field in fields])],
                x_label="Absolute error",
                width=500,
                x_max=x_max,
            ),
            ["matched", "random"],
        )

    def pair(fields: tuple[str, ...]) -> str:
        present = [row for row in (*matched_rows, *random_rows) if row["field"] in fields]
        limit = nice_limit(max(max(row["mean"], row["p50"]) for row in present))
        return grid(panel("Mean", chart(fields, "mean", limit)), panel("Median", chart(fields, "p50", limit)))

    analyzer = ("official_sr",) + tuple(row["field"] for row in matched_rows if str(row["field"]).startswith("analyzer_"))
    msd = tuple(row["field"] for row in matched_rows if str(row["field"]).startswith("msd_"))
    return "<h3>Chart analysis</h3>" + pair(analyzer) + "<h3>MSD</h3>" + pair(msd)


def _match_series(matched_summary: list[dict], random_summary: list[dict], level: str, metric: str):
    categories = []
    left, right = [], []
    band = {"matched": ([], [], []), "random": ([], [], [])}
    for left_row, right_row in zip(matched_summary, random_summary):
        if left_row["level"] != level or left_row["metric"] != metric:
            continue
        categories.append(f"{SIDE_LABEL[left_row['side']]} · {KEY_LABEL.get(left_row['key'], left_row['key'])}")
        left.append(left_row["mean"])
        right.append(right_row["mean"])
        for name, row in (("matched", left_row), ("random", right_row)):
            band[name][0].append(row["p10"])
            band[name][1].append(row["p50"])
            band[name][2].append(row["p90"])
    return categories, left, right, band


def match_mean_chart(matched_summary: list[dict], random_summary: list[dict], level: str, metric: str, x_label: str) -> str:
    categories, left, right, _band = _match_series(matched_summary, random_summary, level, metric)
    return frame(
        svg_horizontal(categories, [("matched", left), ("random", right)], x_label=x_label, width=500, x_max=1),
        ["matched", "random"],
    )


def match_range_chart(matched_summary: list[dict], random_summary: list[dict], level: str) -> str:
    panels = []
    for side, title in (("lane_aware", "Lane-aware"), ("lane_agnostic", "Lane-agnostic")):
        categories = []
        matched_band = ([], [], [])
        random_band = ([], [], [])
        for left_row, right_row in zip(matched_summary, random_summary):
            if left_row["level"] != level or left_row["metric"] != "f1" or left_row["side"] != side:
                continue
            categories.append(KEY_LABEL.get(left_row["key"], left_row["key"]))
            for band, row in ((matched_band, left_row), (random_band, right_row)):
                band[0].append(row["p10"])
                band[1].append(row["p50"])
                band[2].append(row["p90"])
        panels.append(
            panel(
                title,
                frame(
                    svg_range(
                        categories,
                        [("matched", *matched_band), ("random", *random_band)],
                        y_label="F1",
                        width=500,
                        height=260,
                        y_max=1,
                    ),
                    ["matched", "random"],
                ),
            )
        )
    return grid(*panels)


def motif_top_charts(rows: list[list[object]]) -> str:
    """rows are scope, rank, motif, length, human, matched, random, matched-human, already formatted or numeric."""
    parts = ["<h3>Largest frequency gaps</h3>", "<p class=\"lead\">Each scope shows the 15 motifs with the largest |matched − human|. Positive means the generated charts use it more often.</p>"]
    by_scope: dict[str, list[list[object]]] = {}
    for row in rows:
        by_scope.setdefault(str(row[0]), []).append(row)
    for scope, scope_rows in by_scope.items():
        chosen = scope_rows[:15]
        labels = [str(row[2]) for row in chosen]
        matched_delta = [float(row[7]) for row in chosen]
        random_delta = [float(row[6]) - float(row[4]) for row in chosen]
        parts.append(
            panel(
                scope,
                frame(
                    svg_horizontal(
                        labels,
                        [("matched − human", matched_delta), ("random − human", random_delta)],
                        x_label="Frequency difference",
                        width=980,
                        diverging=True,
                    ),
                    ["matched − human", "random − human"],
                ),
            )
        )
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the v0 baseline corpus report")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--motif-vocab", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    data_dir = run_dir / "report" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "matched": run_dir / "exp1_matched_constrained.json",
        "random": run_dir / "exp2_random_constrained.json",
        "unconstrained": run_dir / "exp3_unconstrained.json",
        "teacher_matched": run_dir / "teacher_matched_full.json",
        "teacher_random": run_dir / "teacher_random_full.json",
        "motif_vocab": args.motif_vocab,
        "driver_log": run_dir / "driver.log",
    }
    for path in inputs.values():
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    matched_report = load_json(inputs["matched"])
    random_report = load_json(inputs["random"])
    unconstrained_report = load_json(inputs["unconstrained"])
    teacher_matched = load_json(inputs["teacher_matched"])
    teacher_random = load_json(inputs["teacher_random"])
    matched = constrained_row(matched_report, "matched_full")
    random_row = constrained_row(random_report, "random_full")
    paths = [chart["osu"] for chart in matched["charts"]]
    random_paths = [chart["osu"] for chart in random_row["charts"]]
    if set(paths) != set(random_paths):
        raise SystemExit("matched and random target chart sets differ")

    vocab_json = load_json(inputs["motif_vocab"])
    vocab = load_ordered_motif_vocab(vocab_json)
    log(f"recomputing human test-split stats for {len(paths)} charts")
    human_stats, human_motifs = human_corpus(paths, vocab)
    log("checking motif distance against the stored result")
    check_motif_distance(matched["motif_distance"], matched["motif_distribution"], human_motifs)
    check_motif_distance(random_row["motif_distance"], random_row["motif_distribution"], human_motifs)

    matched_summary = match_summary(matched["charts"])
    random_summary = match_summary(random_row["charts"])
    matched_adherence = adherence_summary(matched["charts"])
    random_adherence = adherence_summary(random_row["charts"])
    write_csv(
        data_dir / "matching_summary.csv",
        ["scenario", "level", "side", "key", "metric", "mean", "p10", "p50", "p90"],
        [
            [scenario, row["level"], row["side"], row["key"], row["metric"], row["mean"], row["p10"], row["p50"], row["p90"]]
            for scenario, rows in (("matched_full", matched_summary), ("random_full", random_summary))
            for row in rows
        ],
    )
    match_header = ["osu", "scenario", "level", "side", "key", "tp", "fp", "fn", "precision", "recall", "f1"]
    match_rows = []
    for scenario, charts in (("matched_full", matched["charts"]), ("random_full", random_row["charts"])):
        for chart in charts:
            for level, keys in MATCH_TREE.items():
                for side in SIDES:
                    for key in keys:
                        cell = chart["match"][level][side][key]
                        match_rows.append(
                            [chart["osu"], scenario, level, side, key, cell["tp"], cell["fp"], cell["fn"], cell["precision"], cell["recall"], cell["f1"]]
                        )
    write_csv(data_dir / "matching_per_chart.csv", match_header, match_rows)
    write_csv(
        data_dir / "adherence_summary.csv",
        ["scenario", "field", "n", "missing", "mean", "p10", "p50", "p90"],
        [
            [scenario, row["field"], row["n"], row["missing"], row["mean"], row["p10"], row["p50"], row["p90"]]
            for scenario, rows in (("matched_full", matched_adherence), ("random_full", random_adherence))
            for row in rows
        ],
    )
    adherence_rows = []
    for scenario, charts in (("matched_full", matched["charts"]), ("random_full", random_row["charts"])):
        for chart in charts:
            adherence_rows.append([chart["osu"], scenario, *[chart["adherence"][field] for field in ADHERENCE_FIELDS]])
    write_csv(data_dir / "adherence_per_chart.csv", ["osu", "scenario", *ADHERENCE_FIELDS], adherence_rows)

    def dump_grid(name: str, section: str) -> None:
        header = ["lane", "state", "bin", "metric", "human", "matched", "random"]
        rows = []
        sources = {
            "human": human_stats[section],
            "matched": matched["statistics"][section],
            "random": random_row["statistics"][section],
        }
        sample = sources["human"]["pooled_frequency"]
        if section == "row":
            keys = sorted(set(sample) | set(sources["matched"]["pooled_frequency"]) | set(sources["random"]["pooled_frequency"]))
            for metric in ("avg_count", "pooled_frequency"):
                for key in keys:
                    rows.append(
                        [
                            "",
                            "",
                            key,
                            metric,
                            sources["human"][metric].get(key, 0),
                            sources["matched"][metric].get(key, 0),
                            sources["random"][metric].get(key, 0),
                        ]
                    )
        elif section == "note":
            for lane in sample:
                for kind in sample[lane]:
                    for metric in ("avg_count", "pooled_frequency"):
                        rows.append(
                            [
                                lane,
                                kind,
                                "",
                                metric,
                                sources["human"][metric][lane][kind],
                                sources["matched"][metric][lane][kind],
                                sources["random"][metric][lane][kind],
                            ]
                        )
        elif section == "hold":
            lanes = list(dict.fromkeys([*sample, *sources["matched"]["pooled_frequency"], *sources["random"]["pooled_frequency"]]))
            for lane in lanes:
                bins = set(sample.get(lane, {}))
                bins |= set(sources["matched"]["pooled_frequency"].get(lane, {}))
                bins |= set(sources["random"]["pooled_frequency"].get(lane, {}))
                for bin_name in sorted(bins, key=lambda item: int(item)):
                    for metric in ("avg_count", "pooled_frequency"):
                        rows.append(
                            [
                                lane,
                                "",
                                bin_name,
                                metric,
                                sources["human"][metric][lane].get(bin_name, 0),
                                sources["matched"][metric][lane].get(bin_name, 0),
                                sources["random"][metric][lane].get(bin_name, 0),
                            ]
                        )
        else:
            for lane in sample:
                for state in sample[lane]:
                    bins = sorted(sample[lane][state], key=lambda item: int(item) if item.isdigit() else item)
                    for bin_name in bins:
                        for metric in ("avg_count", "pooled_frequency"):
                            rows.append(
                                [
                                    lane,
                                    state,
                                    bin_name,
                                    metric,
                                    sources["human"][metric][lane][state].get(bin_name, 0),
                                    sources["matched"][metric][lane][state].get(bin_name, 0),
                                    sources["random"][metric][lane][state].get(bin_name, 0),
                                ]
                            )
        write_csv(data_dir / f"{name}.csv", header, rows)

    for section in ("tick", "snap", "note", "row", "hold"):
        dump_grid(section, section)

    motif_top_rows = []
    length_rows = []
    mae_length_rows = []
    mae_by_scope: dict[str, list[dict]] = {}
    for scope in ("lane", "hand", "row"):
        texts = vocab.texts[scope]
        human_f = human_motifs.frequency(scope)
        matched_f = matched["motif_distribution"]["scopes"][scope]["frequency"]
        random_f = random_row["motif_distribution"]["scopes"][scope]["frequency"]
        human_avg = human_motifs.avg_count(scope)
        matched_avg = matched["motif_distribution"]["scopes"][scope]["avg_count"]
        random_avg = random_row["motif_distribution"]["scopes"][scope]["avg_count"]
        write_csv(
            data_dir / f"motif_{scope}.csv",
            ["motif", "length", "human_frequency", "matched_frequency", "random_frequency", "human_avg_count", "matched_avg_count", "random_avg_count"],
            [
                [text, motif_length(parse_motif(text)), human_f[i], matched_f[i], random_f[i], human_avg[i], matched_avg[i], random_avg[i]]
                for i, text in enumerate(texts)
            ],
        )
        order = sorted(range(len(texts)), key=lambda index: (-abs(matched_f[index] - human_f[index]), texts[index]))
        for rank, index in enumerate(order[:30], start=1):
            motif_top_rows.append(
                [
                    scope,
                    rank,
                    texts[index],
                    motif_length(parse_motif(texts[index])),
                    fmt(human_f[index], 6),
                    fmt(matched_f[index], 6),
                    fmt(random_f[index], 6),
                    fmt(matched_f[index] - human_f[index], 6),
                ]
            )
        human_len = length_counts(human_f, human_motifs.observations[scope], texts)
        matched_len = length_counts(matched_f, matched["motif_distribution"]["scopes"][scope]["observations"], texts)
        random_len = length_counts(random_f, random_row["motif_distribution"]["scopes"][scope]["observations"], texts)
        for length in sorted(set(human_len) | set(matched_len) | set(random_len)):
            length_rows.append([scope, length, human_len.get(length, 0), matched_len.get(length, 0), random_len.get(length, 0)])
        matched_parts = mae_by_length(matched_f, human_f, texts)
        random_parts = mae_by_length(random_f, human_f, texts)
        if abs(sum(part["contribution"] for part in matched_parts) - matched["motif_distance"][scope]["mae"]) > 1e-8:
            raise SystemExit(f"{scope} matched length contributions do not add up to the stored MAE")
        if abs(sum(part["contribution"] for part in random_parts) - random_row["motif_distance"][scope]["mae"]) > 1e-8:
            raise SystemExit(f"{scope} random length contributions do not add up to the stored MAE")
        random_by_length = {part["length"]: part for part in random_parts}
        scope_rows = []
        for part in matched_parts:
            other = random_by_length[part["length"]]
            scope_rows.append(
                {
                    "length": part["length"],
                    "motifs": part["motifs"],
                    "matched_contribution": part["contribution"],
                    "matched_share": part["share"],
                    "random_contribution": other["contribution"],
                    "random_share": other["share"],
                }
            )
            mae_length_rows.append(
                [
                    scope,
                    part["length"],
                    part["motifs"],
                    part["contribution"],
                    part["share"],
                    other["contribution"],
                    other["share"],
                ]
            )
        mae_by_scope[scope] = scope_rows
    write_csv(
        data_dir / "motif_top_abs_delta.csv",
        ["scope", "rank", "motif", "length", "human_frequency", "matched_frequency", "random_frequency", "matched_minus_human"],
        motif_top_rows,
    )
    write_csv(data_dir / "motif_length_counts.csv", ["scope", "length", "human", "matched", "random"], length_rows)
    write_csv(
        data_dir / "motif_mae_by_length.csv",
        ["scope", "length", "motifs", "matched_contribution", "matched_share", "random_contribution", "random_share"],
        mae_length_rows,
    )

    diversity = matched_report["diversity"]["matched_full"]
    diversity_rows = []
    aware_f1 = []
    agnostic_f1 = []
    for item in diversity:
        for pair in item["matching"]:
            aware = pair["match"]["note"]["lane_aware"]["all_notes"]["f1"]
            agnostic = pair["match"]["note"]["lane_agnostic"]["all_notes"]["f1"]
            aware_f1.append(aware)
            agnostic_f1.append(agnostic)
            motif_cells = {scope: pair_motif["distance"][scope] for scope in ("lane", "hand", "row") for pair_motif in item["motif"] if pair_motif["i"] == pair["i"] and pair_motif["j"] == pair["j"]}
            diversity_rows.append(
                [
                    item["osu"],
                    pair["i"],
                    pair["j"],
                    aware,
                    agnostic,
                    motif_cells["lane"]["mae"],
                    motif_cells["hand"]["mae"],
                    motif_cells["row"]["mae"],
                    motif_cells["lane"]["spearman"],
                    motif_cells["hand"]["spearman"],
                    motif_cells["row"]["spearman"],
                ]
            )
    write_csv(
        data_dir / "diversity_pairs.csv",
        ["osu", "i", "j", "note_f1_lane_aware", "note_f1_lane_agnostic", "lane_mae", "hand_mae", "row_mae", "lane_spearman", "hand_spearman", "row_spearman"],
        diversity_rows,
    )
    unc = unconstrained_report["scenarios"]["matched_full"]["unconstrained"][0]
    write_csv(
        data_dir / "unconstrained_charts.csv",
        ["osu", "validity", "missing_hold_tail_count"],
        [
            [chart["osu"], chart["validity"], sum(1 for issue in chart["decode_issues"] if issue.get("kind") == "missing_hold_tail")]
            for chart in unc["charts"]
        ],
    )

    matched_tail = hold_tail_from_charts(matched["charts"])
    random_tail = hold_tail_from_charts(random_row["charts"])
    stored_tail = matched["missing_hold_tail"]
    if matched_tail["instances"] != stored_tail["missing_hold_tail_count"]:
        raise SystemExit("matched hold-tail instance count does not match the stored report")
    if abs(matched_tail["charts_with"] / matched_tail["charts"] - stored_tail["missing_hold_tail_rate"]) > 1e-12:
        raise SystemExit("matched hold-tail rate does not match the stored report")
    random_stored = random_row["missing_hold_tail"]
    if random_tail["instances"] != random_stored["missing_hold_tail_count"]:
        raise SystemExit("random hold-tail instance count does not match the stored report")
    if abs(random_tail["charts_with"] / random_tail["charts"] - random_stored["missing_hold_tail_rate"]) > 1e-12:
        raise SystemExit("random hold-tail rate does not match the stored report")

    hold_tail_rows = []
    for scenario, charts in (("matched_full", matched["charts"]), ("random_full", random_row["charts"])):
        for chart in charts:
            count = sum(1 for issue in chart["decode_issues"] if issue.get("kind") == "missing_hold_tail")
            hold_tail_rows.append([chart["osu"], scenario, "", count])
    diversity_tail_summary = []
    for seed in range(5):
        instances = 0
        charts_with = 0
        for item in diversity:
            count = sum(1 for issue in item["decode_issues"][seed] if issue.get("kind") == "missing_hold_tail")
            instances += count
            charts_with += int(count > 0)
            hold_tail_rows.append([item["osu"], "diversity", seed, count])
        diversity_tail_summary.append([seed, len(diversity), charts_with, instances, fmt(charts_with / len(diversity))])
    write_csv(
        data_dir / "missing_hold_tail_charts.csv",
        ["osu", "scenario", "seed", "missing_hold_tail_count"],
        hold_tail_rows,
    )

    tick_names = list(human_stats["tick"]["pooled_frequency"]["all_lanes"]["all_event"])
    row_keys = sorted(
        human_stats["row"]["pooled_frequency"],
        key=lambda key: human_stats["row"]["pooled_frequency"][key],
        reverse=True,
    )
    hold_keys = sorted(
        human_stats["hold"]["pooled_frequency"]["all_lanes"],
        key=lambda key: human_stats["hold"]["pooled_frequency"]["all_lanes"][key],
        reverse=True,
    )

    teacher_headers = ["Metric", "matched", "random"]
    teacher_keys = (
        ("charts", "Charts", None),
        ("loss", "mean token NLL", 4),
        ("token_acc", "token accuracy", 4),
        ("ppl", "PPL", 4),
        ("legal_top1", "legal top-1", 6),
        ("legal_probability", "legal probability", 6),
    )
    teacher_rows = []
    left = teacher_matched["scenarios"]["matched_full"]
    right = teacher_random["scenarios"]["random_full"]
    for key, label, digits in teacher_keys:
        teacher_rows.append([label, left[key] if digits is None else fmt(left[key], digits), right[key] if digits is None else fmt(right[key], digits)])
    for key, label in (
        ("bar_acc", "bar accuracy"),
        ("pos_acc", "pos accuracy"),
        ("row_exact_acc", "row exact accuracy"),
        ("row_lane_acc", "row lane accuracy"),
        ("eos_acc", "EOS accuracy"),
        ("tap_recall", "tap recall"),
        ("hold_start_recall", "hold start recall"),
        ("hold_end_recall", "hold end recall"),
        ("other_acc", "other accuracy"),
    ):
        teacher_rows.append([label, fmt(left["split_token_acc"][key], 4), fmt(right["split_token_acc"][key], 4)])
    for key in ("bar", "pos", "row", "eos", "other"):
        teacher_rows.append([f"{key} NLL", fmt(left["split_nll"][key], 4), fmt(right["split_nll"][key], 4)])

    distance_rows = []
    for scenario, block in (("matched_full", matched["motif_distance"]), ("random_full", random_row["motif_distance"])):
        for scope, body in block.items():
            distance_rows.append(
                [
                    scenario,
                    scope,
                    fmt(body["mae"], 6),
                    fmt(body["rmse"], 6),
                    fmt(body["spearman"]),
                    body["observations_generated"],
                    body["observations_human"],
                    body["largest_abs_deviation"]["motif"],
                    fmt(body["largest_abs_deviation"]["delta"], 6),
                    body["largest_generated_excess"]["motif"],
                    fmt(body["largest_generated_excess"]["delta"], 6),
                ]
            )
    matched_distance = {row[1]: row for row in distance_rows if row[0] == "matched_full"}
    random_distance = {row[1]: row for row in distance_rows if row[0] == "random_full"}
    motif_table = table(
        ["scope", "matched MAE", "matched RMSE", "matched Spearman", "random MAE", "random RMSE", "random Spearman"],
        [
            [scope, matched_distance[scope][2], matched_distance[scope][3], matched_distance[scope][4], random_distance[scope][2], random_distance[scope][3], random_distance[scope][4]]
            for scope in ("lane", "hand", "row")
        ],
    )
    hold_columns = [
        [
            human_stats["hold"]["pooled_frequency"]["all_lanes"][key],
            matched["statistics"]["hold"]["pooled_frequency"]["all_lanes"].get(key, 0),
            random_row["statistics"]["hold"]["pooled_frequency"]["all_lanes"].get(key, 0),
        ]
        for key in hold_keys[:15]
    ]
    context = {
        "lede": (
            "Checkpoint step_200000, test split of 1,487 charts. Sampling uses temperature 0.8, top-p 0.95, top-k 50, and seed 0. "
            "Matched uses the target chart's condition; random uses a random donor. The reference is the same set of human charts."
        ),
        "teacher_html": (
            "<p class=\"lead\">Next-token prediction on the human prefix. Matched uses the target chart's condition; random uses a random donor. PPL = exp(mean token NLL).</p>"
            + "<h3>Overall</h3>"
            + table(teacher_headers, teacher_rows[:6])
            + "<h3>Per-token accuracy</h3>"
            + table(teacher_headers, teacher_rows[6:15])
            + "<h3>Per-token NLL</h3>"
            + table(teacher_headers, teacher_rows[15:])
        ),
        "validity_html": (
            "<p class=\"lead\">Constrained sampling produced 1,487 legal charts. An open hold is a lane still held when decoding ends. The count is lanes; the rate is charts with at least one.</p>"
            + table(
                ["Sampling", "Charts", "Charts with an open hold", "Open lanes", "Rate"],
                [
                    ["matched", matched_tail["charts"], matched_tail["charts_with"], matched_tail["instances"], fmt(matched_tail["charts_with"] / matched_tail["charts"])],
                    ["random", random_tail["charts"], random_tail["charts_with"], random_tail["instances"], fmt(random_tail["charts_with"] / random_tail["charts"])],
                ],
            )
            + "<h3>Unconstrained</h3>"
            + "<p class=\"lead\">Same matched condition, 300 charts, constraints off. This block is validity only.</p>"
            + table(
                ["Outcome", "Charts", "Rate"],
                [
                    ["valid", sum(chart["validity"] == "valid" for chart in unc["charts"]), fmt(unc["valid_rate"])],
                    ["illegal token", sum(chart["validity"] == "illegal_token" for chart in unc["charts"]), fmt(unc["illegal_token_rate"])],
                    ["truncated", sum(chart["validity"] == "truncated" for chart in unc["charts"]), fmt(unc["truncated_rate"])],
                ],
            )
        ),
        "match_html": (
            "<p class=\"lead\">Each generated chart is matched to its human chart. Bars are the mean over 1,487 charts. The F1 whisker runs from p10 to p90; the dot is the median.</p>"
            + "<h3>F1</h3>"
            + grid(
                panel("Events", match_mean_chart(matched_summary, random_summary, "event", "f1", "Mean F1")),
                panel("Notes", match_mean_chart(matched_summary, random_summary, "note", "f1", "Mean F1")),
            )
            + "<h3>F1 across charts</h3>"
            + panel("Events", match_range_chart(matched_summary, random_summary, "event"))
            + panel("Notes", match_range_chart(matched_summary, random_summary, "note"))
            + "<h3>Precision</h3>"
            + grid(
                panel("Events", match_mean_chart(matched_summary, random_summary, "event", "precision", "Mean precision")),
                panel("Notes", match_mean_chart(matched_summary, random_summary, "note", "precision", "Mean precision")),
            )
            + "<h3>Recall</h3>"
            + grid(
                panel("Events", match_mean_chart(matched_summary, random_summary, "event", "recall", "Mean recall")),
                panel("Notes", match_mean_chart(matched_summary, random_summary, "note", "recall", "Mean recall")),
            )
        ),
        "adherence_html": (
            "<p class=\"lead\">Absolute error between each generated chart and its condition, in the field's own units. All 1,487 charts have a value.</p>"
            + adherence_charts(matched_adherence, random_adherence)
        ),
        "distribution_html": (
            "<p class=\"lead\">Human is the 1,487 target charts. Matched and random are the two samples.</p>"
            + "<h3>Notes per lane</h3>"
            + "<p class=\"lead\">Counts are the mean per chart. Shares use every note on all four lanes as the denominator.</p>"
            + note_charts(human_stats, matched["statistics"], random_row["statistics"])
            + "<h3>Snap</h3>"
            + "<p class=\"lead\">All four lanes combined, from a whole beat down to 1/48. All events and tap share one axis; hold head and hold tail share another, because holds are a much smaller share.</p>"
            + snap_charts(human_stats, matched["statistics"], random_row["statistics"])
            + "<h3>Position within the beat</h3>"
            + "<p class=\"lead\">The axis is the tick remainder within a beat, 0 through 47. All events and tap share one axis; hold head and hold tail share another.</p>"
            + tick_charts(human_stats, matched["statistics"], random_row["statistics"], tick_names)
            + "<h3>Hold length</h3>"
            + "<p class=\"lead\">The 15 durations with the highest share in the human charts. The unit is ticks; one beat is 48.</p>"
            + _compare_bars(hold_keys[:15], hold_columns, "Share", width=980, height=280)
            + "<h3>Row patterns</h3>"
            + "<p class=\"lead\">The 12 rows most common in the human charts. The denominator is how often a row occurs.</p>"
            + frame(
                svg_horizontal(
                    row_keys[:12],
                    [
                        ("human", [human_stats["row"]["pooled_frequency"][key] for key in row_keys[:12]]),
                        ("matched", [matched["statistics"]["row"]["pooled_frequency"].get(key, 0) for key in row_keys[:12]]),
                        ("random", [random_row["statistics"]["row"]["pooled_frequency"].get(key, 0) for key in row_keys[:12]]),
                    ],
                    x_label="Share",
                    width=980,
                ),
                ["human", "matched", "random"],
            )
        ),
        "motif_html": (
            "<p class=\"lead\">Vocabulary size 50,000. Frequency is the motif's share of event positions in that scope.</p>"
            + motif_table
            + mae_length_section(mae_by_scope)
            + motif_top_charts(motif_top_rows)
        ),
        "diversity_html": (
            "<p class=\"lead\">100 charts, 5 seeds each, every pair compared: "
            + str(len(diversity_rows))
            + " pairs. Identical pairs: "
            + str(sum(item["exact"]["exact_pair_count"] for item in diversity))
            + ".</p>"
            + grid(
                panel("Lane-aware, all notes F1", svg_hist(aware_f1, title="Pairs", x_min=0, x_max=1)),
                panel("Lane-agnostic, all notes F1", svg_hist(agnostic_f1, title="Pairs", x_min=0, x_max=1)),
            )
            + table(
                ["", "Lane-aware", "Lane-agnostic"],
                [
                    ["Mean", fmt(mean(aware_f1)), fmt(mean(agnostic_f1))],
                    ["p10", fmt(percentile(aware_f1, 0.1)), fmt(percentile(agnostic_f1, 0.1))],
                    ["p50", fmt(percentile(aware_f1, 0.5)), fmt(percentile(agnostic_f1, 0.5))],
                    ["p90", fmt(percentile(aware_f1, 0.9)), fmt(percentile(agnostic_f1, 0.9))],
                ],
            )
            + "<h3>Open holds on these 100 charts</h3>"
            + table(["Seed", "Charts", "Charts with an open hold", "Open lanes", "Rate"], diversity_tail_summary)
        ),
    }
    report_path = run_dir / "report" / "index.html"
    report_path.write_text(build_html(context), encoding="utf-8")
    manifest = {
        "report": str(report_path),
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "human_charts": len(paths),
        "motif_distance_check": "passed",
    }
    (run_dir / "report" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
