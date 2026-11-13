"""Chart-level distributions, paired uncertainty, and joint-condition responses."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
import math

import numpy as np

from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS

BOOTSTRAP_SEED = 20261106
BOOTSTRAP_DRAWS = 2000
CONFIDENCE_LEVEL = 0.95
QUANTILES = (0.1, 0.5, 0.9)


def mean_available(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return sum(observed) / len(observed) if observed else None


def describe(values: Sequence[float | None]) -> dict:
    data = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not data.size:
        return {"count": 0, **dict.fromkeys(("mean", "p10", "p50", "p90", "min", "max"))}
    p10, p50, p90 = np.quantile(data, QUANTILES).tolist()
    return {
        "count": int(data.size), "mean": float(data.mean()),
        "p10": p10, "p50": p50, "p90": p90,
        "min": float(data.min()), "max": float(data.max()),
    }


def chart_values(rows: Sequence[dict], value: Callable[[dict], float | None]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        number = value(row)
        if number is not None and math.isfinite(number):
            grouped[row["relpath"]].append(number)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def mean_interval(values: Sequence[float]) -> dict:
    """Resample independent charts, never pretend the five seeds are five charts."""
    data = np.asarray(values, dtype=float)
    result = {"charts": int(data.size), "mean": None, "low": None, "high": None}
    if not data.size:
        return result
    result["mean"] = float(data.mean())
    if data.size < 2:
        return result
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = data[rng.integers(data.size, size=(BOOTSTRAP_DRAWS, data.size))].mean(axis=1)
    tail = (1 - CONFIDENCE_LEVEL) / 2
    result["low"], result["high"] = np.quantile(means, (tail, 1 - tail)).tolist()
    return result


def paired_difference(left: dict[str, float], right: dict[str, float]) -> dict:
    """Right minus left, paired by chart. Missing charts stay missing."""
    common = sorted(left.keys() & right.keys())
    return {
        **mean_interval([right[key] - left[key] for key in common]),
        "left_only": len(left.keys() - right.keys()),
        "right_only": len(right.keys() - left.keys()),
    }


def regression_slope(x, y, *, center=False) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not x.size:
        return None
    if center:
        x, y = x - x.mean(), y - y.mean()
    return float(x @ y / (x @ x)) if x @ x else None


def condition_responses(rows: Sequence[dict], *, seed: int) -> dict:
    grouped = defaultdict(dict)
    for row in rows:
        if row["mode"] == "constrained" and row["seed"] == seed:
            grouped[row["relpath"]][row["scenario"]] = row
    pairs = []
    self_draws = 0
    for path, group in grouped.items():
        if not {"matched_full", "random_full"} <= group.keys():
            continue
        full, random = group["matched_full"], group["random_full"]
        if random["condition"]["donor"] == path:
            self_draws += 1
        else:
            pairs.append((path, full, random))
    fields = {}
    for field in COND_SOURCE_FIELDS:
        changes = []
        missing = unchanged = 0
        for path, full, random in pairs:
            target = [row["condition"]["meta"].get(field) for row in (full, random)]
            generated = [row["generated_meta"].get(field) for row in (full, random)]
            if any(v is None or not math.isfinite(v) for v in target + generated):
                missing += 1
                continue
            dx, dy = target[1] - target[0], generated[1] - generated[0]
            if dx == 0:
                unchanged += 1
            else:
                changes.append({"relpath": path, "target_delta": dx, "generated_delta": dy})
        x = np.asarray([row["target_delta"] for row in changes])
        y = np.asarray([row["generated_delta"] for row in changes])
        fields[field] = {
            "pairs": len(changes), "missing": missing, "unchanged_target": unchanged,
            "slope": regression_slope(x, y),
            "direction_matches": int((x * y > 0).sum()),
            "direction_rate": float((x * y > 0).mean()) if x.size else None,
            "mean_target_change": float(np.abs(x).mean()) if x.size else None,
            "mean_generated_change": float(np.abs(y).mean()) if x.size else None,
            "changes": changes,
        }
    return {
        "seed": seed, "pairs": len(pairs), "self_draws_excluded": self_draws,
        "identical_sequences": sum(a["tokens_sha256"] == b["tokens_sha256"] for _, a, b in pairs),
        "fields": fields,
        "interpretation": "Joint condition replacement; not an isolated-field causal effect.",
    }
