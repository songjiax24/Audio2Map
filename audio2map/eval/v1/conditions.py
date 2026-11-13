"""Frozen scenario conditions. Unavailable or dropped fields stay missing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.mask import frozen_partial_mask
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS, build_cond

SCENARIOS = ("matched_full", "random_full", "matched_partial", "random_partial", "none")
UNCONSTRAINED_SCENARIOS = ("matched_full", "matched_partial", "none")
DIVERSITY_SEEDS = (0, 1, 2, 3, 4)


def scenario_cases(
    mode: str, scenario: str, seeds: Sequence[int],
) -> Iterator[tuple[str, int | None]]:
    if mode in ("teacher", "all"):
        yield "teacher", None
    for seed in seeds if scenario == "matched_full" else seeds[:1]:
        if mode in ("constrained", "all"):
            yield "constrained", seed
        if (
            mode in ("unconstrained", "all")
            and scenario in UNCONSTRAINED_SCENARIOS
            and seed == seeds[0]
        ):
            yield "unconstrained", seed


@dataclass(frozen=True, slots=True)
class ScenarioCondition:
    meta: ChartMeta
    donor: str | None
    mask: tuple[int, ...]

    def as_dict(self) -> dict:
        return {"meta": self.meta.to_dict(), "donor": self.donor, "mask": list(self.mask)}


def scenario_condition(
    target: ChartMeta,
    donor: ChartMeta,
    *,
    scenario: str,
    seed: int,
    key: str,
    donor_path: str,
) -> ScenarioCondition:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown v1 scenario: {scenario}")
    random = scenario.startswith("random_")
    source = donor if random else target
    keep = build_cond(source).m_real
    if scenario == "none":
        keep = keep * 0.0
    elif scenario.endswith("_partial"):
        keep = frozen_partial_mask(keep, seed=seed, key=f"{key}\n{scenario}")
    fields = {
        name: getattr(source, name) if keep[index] else None
        for index, name in enumerate(COND_SOURCE_FIELDS)
    }
    return ScenarioCondition(
        replace(source, **fields),
        donor_path if random else None,
        tuple(int(value) for value in keep),
    )
