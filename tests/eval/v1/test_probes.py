import numpy as np
import pytest

from audio2map.dataset.v1.example import ChartAudio
from audio2map.eval.v1.probes import _control_summary, isolated_condition, shift_audio
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import build_cond


def test_audio_shift_preserves_reference_clock_and_disables_stale_mel_cache():
    pcm = np.arange(16, dtype=np.float32)
    audio = ChartAudio((), (), pcm, t_audio_ms=123, audio_key="original")
    shifted, amount = shift_audio(audio, 5)
    assert amount == 4
    assert shifted.audio_key is None
    assert shifted.t_audio_ms == audio.t_audio_ms
    assert shifted.notes is audio.notes and shifted.redlines is audio.redlines
    np.testing.assert_array_equal(shifted.pcm, np.roll(pcm, 4))
    np.testing.assert_array_equal(audio.pcm, np.arange(16))
    assert shift_audio(audio, 0)[0].audio_key is None


def test_isolated_request_masks_correlated_conditions_and_summary_averages_seeds():
    meta = ChartMeta("a", official_sr=3, rework_sr=4, msd_overall=20, analyzer_ln_percent=0.7)
    request = isolated_condition(meta, "official_sr", 6)
    assert request.official_sr == 6 and request.msd_overall is None
    assert build_cond(request).m_real.sum() == 1
    assert build_cond(isolated_condition(meta)).m_real.sum() == 0
    rows = [
        {"field": "official_sr", "requested": level, "relpath": "a", "seed": seed,
         "generated_meta": {"official_sr": level / 2 + seed}}
        for level in (2.0, 4.0, 6.0) for seed in (0, 1)
    ]
    result = _control_summary(rows, "official_sr")
    assert result["levels"]["2.0"]["chart_mean"]["charts"] == 1
    assert result["charts"]["a"]["slope"] == pytest.approx(0.5)
    assert result["charts"]["a"]["nondecreasing"]
