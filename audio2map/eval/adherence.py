"""Name-based condition adherence in original ``ChartMeta`` units.

The model may still receive ``canonical_bpm_norm``. That field is not an
adherence target. Missing values stay missing.
"""

from __future__ import annotations

from audio2map.features.cond import USER_COND_SOURCE_FIELDS, ChartMeta


def source_values(
    meta: ChartMeta,
    names: tuple[str, ...] = USER_COND_SOURCE_FIELDS,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in names:
        raw = getattr(meta, name)
        values[name] = None if raw is None else float(raw)
    return values


def condition_mae(
    target: ChartMeta | dict[str, float | None],
    generated: ChartMeta | dict[str, float | None],
    names: tuple[str, ...] = USER_COND_SOURCE_FIELDS,
) -> dict[str, float | None]:
    """Per-name absolute error. A missing side stays missing. No composite score."""
    target_values = source_values(target) if isinstance(target, ChartMeta) else target
    generated_values = source_values(generated) if isinstance(generated, ChartMeta) else generated
    out: dict[str, float | None] = {}
    for name in names:
        left = target_values.get(name)
        right = generated_values.get(name)
        if left is None or right is None:
            out[name] = None
        else:
            out[name] = abs(float(left) - float(right))
    return out
