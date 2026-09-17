#!/usr/bin/env python3
"""Fast tick quantization & snap stats (original BPM, 1/48 grid).

One file read per chart; vectorized numpy; multiprocessing; histogram percentiles.
Training code uses ``CanonicalTiming`` (canonical BPM) — see ``--compare-canonical``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio2map.grid import TICKS_PER_BEAT, canonicalize_bpm
from audio2map.osu.parser import (
    InvalidHitObjectError,
    InvalidTimingPointError,
    is_mania_4k_sections,
    parse_hit_object,
    parse_sections,
    parse_timing_points,
)
from audio2map.osu.timing import beat_length_to_bpm, summarize_timing

TICKS = TICKS_PER_BEAT  # 48

# pos (0..47) -> which 1/n snaps (n=1..24) hit
_SNAP_TABLE = np.zeros((TICKS, 24), dtype=np.uint8)
for _pos in range(TICKS):
    for _n in range(1, 25):
        if (_pos * _n) % TICKS == 0:
            _SNAP_TABLE[_pos, _n - 1] = 1

# abs error histogram: [0,0.25), [0.25,0.5), ... [20,inf)
_ABS_BIN_EDGES = np.array(
    [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 1e9],
    dtype=np.float64,
)


def bpm_band(original_bpm: float, width: int = 10) -> str:
    lo = int(math.floor(original_bpm / width) * width)
    return f"{lo}-{lo + width}"


def _fast_parse_endpoints(text: str) -> tuple[float, int, np.ndarray] | None:
    """Return ``(original_bpm, offset_ms, times_ms)`` or None if ineligible."""
    sections = parse_sections(text)
    if not is_mania_4k_sections(sections):
        return None

    try:
        tps = parse_timing_points(sections.get("[TimingPoints]", []))
    except InvalidTimingPointError:
        return None
    summary = summarize_timing(tps)
    if summary.n_uninherited == 0:
        return None
    if not summary.constant_bpm or not summary.constant_meter or summary.meter != 4:
        return None

    first = next(tp for tp in tps if tp.uninherited)
    original_bpm = beat_length_to_bpm(first.beat_length_ms)
    offset_ms = first.offset_ms

    times: list[int] = []
    try:
        for line in sections.get("[HitObjects]", []):
            note = parse_hit_object(line)
            times.append(note.time_ms)
            if note.end_time_ms is not None:
                times.append(note.end_time_ms)
    except InvalidHitObjectError:
        return None

    if not times:
        return None
    return original_bpm, offset_ms, np.asarray(times, dtype=np.int32)


@dataclass
class Agg:
    charts: int = 0
    n: int = 0
    sum_err: float = 0.0
    sum_err2: float = 0.0
    sum_abs: float = 0.0
    exact: int = 0
    snap: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.int64))
    abs_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(len(_ABS_BIN_EDGES) - 1, dtype=np.int64)
    )
    canon_sum_abs: float = 0.0
    canon_exact: int = 0

    def merge(self, other: Agg) -> None:
        self.charts += other.charts
        self.n += other.n
        self.sum_err += other.sum_err
        self.sum_err2 += other.sum_err2
        self.sum_abs += other.sum_abs
        self.exact += other.exact
        self.snap += other.snap
        self.abs_hist += other.abs_hist
        self.canon_sum_abs += other.canon_sum_abs
        self.canon_exact += other.canon_exact

    def percentile_abs(self, p: float) -> float:
        if self.n == 0:
            return 0.0
        target = self.n * p / 100.0
        cum = 0
        for i, count in enumerate(self.abs_hist):
            cum += count
            if cum >= target:
                lo = _ABS_BIN_EDGES[i]
                hi = _ABS_BIN_EDGES[i + 1]
                return float((lo + hi) / 2.0 if hi < 1e8 else lo)
        return float(_ABS_BIN_EDGES[-2])

    def summary(self) -> dict:
        if self.n == 0:
            return {"charts": self.charts, "endpoints": 0}
        n = self.n
        mean = self.sum_err / n
        var = max(0.0, self.sum_err2 / n - mean * mean)
        hist_labels = []
        for i in range(len(self.abs_hist)):
            lo = _ABS_BIN_EDGES[i]
            hi = _ABS_BIN_EDGES[i + 1]
            label = f"[{lo},{hi})" if hi < 1e8 else f"[{lo},+inf)"
            hist_labels.append(
                {
                    "bin": label,
                    "count": int(self.abs_hist[i]),
                    "ratio": int(self.abs_hist[i]) / n,
                }
            )
        return {
            "charts": self.charts,
            "endpoints": n,
            "error_ms": {
                "mean": mean,
                "std": math.sqrt(var),
            },
            "abs_error_ms": {
                "mean": self.sum_abs / n,
                "p50": self.percentile_abs(50),
                "p90": self.percentile_abs(90),
                "p99": self.percentile_abs(99),
            },
            "exact_on_grid": self.exact / n,
            "abs_error_histogram": hist_labels,
            "snap_ratio_1_over_n": {
                str(i + 1): int(self.snap[i]) / n for i in range(24)
            },
        }


def _stats_from_times(
    times: np.ndarray,
    *,
    original_bpm: float,
    offset_ms: int,
    compare_canonical: bool,
) -> Agg:
    tick_ms = 60000.0 / original_bpm / TICKS
    offset = float(offset_ms)
    t = times.astype(np.float64)
    ticks = np.rint((t - offset) / tick_ms).astype(np.int64)
    q = np.rint(offset + ticks * tick_ms)
    err = t - q
    abs_err = np.abs(err)

    out = Agg(charts=1, n=int(len(t)))
    out.sum_err = float(err.sum())
    out.sum_err2 = float((err * err).sum())
    out.sum_abs = float(abs_err.sum())
    out.exact = int(np.sum(abs_err == 0))

    pos = ticks % TICKS
    pos = np.where(pos < 0, pos + TICKS, pos)
    out.snap = _SNAP_TABLE[pos].sum(axis=0).astype(np.int64)

    out.abs_hist += np.histogram(abs_err, bins=_ABS_BIN_EDGES)[0]

    if compare_canonical:
        canon_bpm, _ = canonicalize_bpm(original_bpm)
        tick_ms_c = 60000.0 / canon_bpm / TICKS
        qc = np.rint(offset + np.rint((t - offset) / tick_ms_c) * tick_ms_c)
        abs_c = np.abs(t - qc)
        out.canon_sum_abs = float(abs_c.sum())
        out.canon_exact = int(np.sum(abs_c == 0))

    return out


def _process_path(path_str: str, *, compare_canonical: bool) -> tuple[str, Agg] | None:
    try:
        text = Path(path_str).read_text(encoding="utf-8", errors="replace")
        parsed = _fast_parse_endpoints(text)
        if parsed is None:
            return None
        bpm, offset_ms, times = parsed
        band = bpm_band(bpm)
        return band, _stats_from_times(
            times,
            original_bpm=bpm,
            offset_ms=offset_ms,
            compare_canonical=compare_canonical,
        )
    except OSError:
        return None


def _collect_paths(root: Path | None, limit: int | None) -> list[str]:
    from audio2map.utils.paths import raw_dir

    root = root or raw_dir()
    paths = [str(p) for p in sorted(root.rglob("*.osu"))]
    if limit:
        paths = paths[:limit]
    return paths


def export_csv(report: dict, csv_path: Path) -> None:
    """Write per-band summary + snap ratios as CSV."""
    import csv

    snap_cols = [f"snap_1_{n}" for n in range(1, 25)]
    hist_bins = [h["bin"] for h in report["global"].get("abs_error_histogram", [])]
    hist_cols = [f"hist_{b.replace('[', '').replace(')', '').replace(',', '_')}" for b in hist_bins]

    rows: list[dict] = []
    for band, data in report["by_original_bpm_band"].items():
        if not data.get("endpoints"):
            continue
        row = {
            "bpm_band": band,
            "charts": data["charts"],
            "endpoints": data["endpoints"],
            "err_mean_ms": round(data["error_ms"]["mean"], 4),
            "err_std_ms": round(data["error_ms"]["std"], 4),
            "abs_err_mean_ms": round(data["abs_error_ms"]["mean"], 4),
            "abs_err_p50_ms": round(data["abs_error_ms"]["p50"], 4),
            "abs_err_p90_ms": round(data["abs_error_ms"]["p90"], 4),
            "abs_err_p99_ms": round(data["abs_error_ms"]["p99"], 4),
            "exact_on_grid": round(data["exact_on_grid"], 6),
        }
        for n in range(1, 25):
            row[f"snap_1_{n}"] = round(data["snap_ratio_1_over_n"][str(n)], 6)
        for h in data.get("abs_error_histogram", []):
            col = f"hist_{h['bin'].replace('[', '').replace(')', '').replace(',', '_')}"
            row[col] = round(h["ratio"], 6)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fast tick quantization stats")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--band-width", type=int, default=10, help="unused (bands fixed at 10)")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--compare-canonical", action="store_true", help="also vs training canonical BPM")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None, help="per-band CSV (default: same stem as --out)")
    args = p.parse_args()

    paths = _collect_paths(args.root, args.limit)
    jobs = args.jobs or max(1, (__import__("os").cpu_count() or 4) - 1)

    bands: dict[str, Agg] = {}
    global_agg = Agg()
    skipped = 0

    print(f"processing {len(paths)} .osu files with {jobs} workers …", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = {
            pool.submit(_process_path, ps, compare_canonical=args.compare_canonical): ps
            for ps in paths
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(paths)} …", file=sys.stderr)
            result = fut.result()
            if result is None:
                skipped += 1
                continue
            band, agg = result
            bands.setdefault(band, Agg()).merge(agg)
            global_agg.merge(agg)

    report = {
        "description": (
            "Original BPM, tick=round((time_ms-offset)/tick_ms), tick_ms=60000/bpm/48, "
            "error=time_ms-quantized_ms; endpoints=tap heads + LN tails"
        ),
        "training_code": (
            "grid.ms_to_tick uses CanonicalTiming.tick_ms (canonical BPM); "
            "encode_notes snaps via ms_to_tick, identity is TickNote"
        ),
        "files_scanned": len(paths),
        "charts_used": global_agg.charts,
        "files_skipped": skipped,
        "endpoints": global_agg.n,
        "global": global_agg.summary(),
        "by_original_bpm_band": {
            k: bands[k].summary()
            for k in sorted(bands, key=lambda s: float(s.split("-")[0]))
        },
    }
    if args.compare_canonical and global_agg.n:
        n = global_agg.n
        report["canonical_bpm_abs_error_ms"] = {
            "mean": global_agg.canon_sum_abs / n,
            "exact_on_grid": global_agg.canon_exact / n,
        }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
        csv_path = args.csv or args.out.with_suffix(".csv")
        export_csv(report, csv_path)
        print(f"wrote {csv_path}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
