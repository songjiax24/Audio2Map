from dataclasses import replace

import numpy as np
import pytest

from audio2map.eval.v1.conditions import scenario_condition
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import COND_SOURCE_FIELDS, build_cond


def condition(target, scenario, donor=None):
    return scenario_condition(
        target,
        donor or target,
        scenario=scenario,
        seed=7,
        key="target.osu",
        donor_path="donor.osu",
    )


def test_full_partial_none_and_missing_fields():
    target = ChartMeta("target", official_sr=0.0, msd_overall=12, pattern_jacks=0.4)
    full = condition(target, "matched_full")
    assert build_cond(full.meta).m_real.sum() == 3
    assert full.meta.official_sr == 0.0
    none = condition(target, "none")
    assert build_cond(none.meta).m_real.sum() == 0
    assert all(getattr(none.meta, name) is None for name in COND_SOURCE_FIELDS)
    partial = condition(target, "matched_partial")
    assert partial.mask == condition(target, "matched_partial").mask
    assert 0 < sum(partial.mask) < 3
    assert np.all(build_cond(partial.meta).m_real <= build_cond(target).m_real)
    assert target.msd_overall == 12


def test_random_targets_the_donor_and_nonfinite_fields_stay_missing():
    target = ChartMeta("target", official_sr=2, msd_overall=12)
    donor = replace(target, osu_path="donor", official_sr=8, msd_overall=float("nan"))
    chosen = condition(target, "random_full", donor)
    assert chosen.donor == "donor.osu"
    assert chosen.meta.official_sr == 8
    assert chosen.meta.msd_overall is None
    assert condition(target, "matched_full", donor).donor is None
    with pytest.raises(ValueError):
        condition(target, "random_none", donor)
