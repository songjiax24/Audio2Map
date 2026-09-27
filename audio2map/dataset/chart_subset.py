"""Pinned chart subset for diversity. Membership is the saved path list.

The seed only draws the list the first time. Later runs load that list, so
v0 and v1 stay on the same charts when the surrounding pool changes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from audio2map.utils.paths import splits_dir


class ChartSubsetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChartSubset:
    seed: int | None
    charts: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"seed": self.seed, "charts": list(self.charts)}

    @classmethod
    def from_dict(cls, data: dict) -> ChartSubset:
        charts = tuple(str(rel) for rel in data.get("charts", []))
        if len(charts) != len(set(charts)):
            raise ChartSubsetError("diversity subset contains duplicate charts")
        seed = data.get("seed")
        return cls(seed=None if seed is None else int(seed), charts=charts)


def default_diversity_subset_path() -> Path:
    return splits_dir() / "diversity_subset.json"


def sample_chart_subset(universe: list[str], count: int, *, seed: int) -> tuple[str, ...]:
    """Draw ``count`` charts from a sorted universe. Input order is ignored."""
    pool = sorted(set(universe))
    if count < 1 or count > len(pool):
        raise ChartSubsetError(
            f"diversity subset count {count} is outside 1..{len(pool)}"
        )
    chosen = random.Random(seed).sample(pool, count)
    return tuple(sorted(chosen))


def save_chart_subset(subset: ChartSubset, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise ChartSubsetError(
            f"diversity subset already exists: {path}; write a new file instead of overwriting"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset.to_dict(), indent=2), encoding="utf-8")


def load_chart_subset(path: Path) -> ChartSubset:
    path = Path(path)
    if not path.is_file():
        raise ChartSubsetError(f"diversity subset missing: {path}")
    return ChartSubset.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_or_create_chart_subset(
    path: Path,
    universe: list[str],
    *,
    count: int = 0,
    seed: int = 0,
    explicit: list[str] | None = None,
) -> ChartSubset:
    """Load a saved subset, or create one from explicit paths or a seeded draw.

    A saved file is not rewritten. Charts in the file must still be in
    ``universe``. Explicit paths, when also passed, must name that same set.
    """
    path = Path(path)
    universe_set = set(universe)
    if path.exists():
        saved = load_chart_subset(path)
        missing = [rel for rel in saved.charts if rel not in universe_set]
        if missing:
            raise ChartSubsetError(
                f"{len(missing)} diversity chart(s) are not in this eval pool: {missing[:20]}"
            )
        if explicit is not None and tuple(sorted(explicit)) != tuple(sorted(saved.charts)):
            raise ChartSubsetError(
                f"explicit diversity charts do not match saved subset {path}"
            )
        if count and count != len(saved.charts):
            raise ChartSubsetError(
                f"--diversity-charts {count} does not match saved subset length {len(saved.charts)}"
            )
        return saved

    if explicit:
        unknown = [rel for rel in explicit if rel not in universe_set]
        if unknown:
            raise ChartSubsetError(
                f"{len(unknown)} explicit diversity chart(s) are not in this eval pool: {unknown[:20]}"
            )
        charts = tuple(sorted(set(explicit)))
        if count and count != len(charts):
            raise ChartSubsetError(
                f"--diversity-charts {count} does not match {len(charts)} explicit charts"
            )
        subset = ChartSubset(seed=None, charts=charts)
    else:
        subset = ChartSubset(seed=seed, charts=sample_chart_subset(universe, count, seed=seed))
    save_chart_subset(subset, path)
    return subset
