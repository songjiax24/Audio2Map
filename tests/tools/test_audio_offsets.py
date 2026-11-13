"""Sign, detector calibration, and unresolved controls for the offset diagnostic."""

import json
import sys

import numpy as np
import pytest

from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.utils.time import MS_PER_SECOND, SAMPLE_RATE
from tools import analyze_audio_offsets as offsets
from tools.analyze_audio_offsets import (
    FFT_SIZES,
    attack_curve,
    calibrate,
    head_times,
    lag_fit,
)


def rendered_attacks(lags):
    rng = np.random.default_rng(10)
    heads = 500.0 + np.cumsum(rng.uniform(300, 550, 120))
    y = np.zeros(round((heads[-1] + 500) / MS_PER_SECOND * SAMPLE_RATE), dtype=np.float32)
    tail = np.arange(round(0.06 * SAMPLE_RATE)) / SAMPLE_RATE
    hit = rng.standard_normal(len(tail)) * np.exp(-tail / 0.02)
    for head, lag in zip(heads, np.broadcast_to(lags, heads.shape), strict=True):
        sample = round((head + lag) / MS_PER_SECOND * SAMPLE_RATE)
        y[sample : sample + len(hit)] += hit
    return heads, y


@pytest.fixture(scope="module")
def calibration():
    return calibrate()


@pytest.mark.parametrize("lag", [-20.0, 0.0, 20.0, 40.0])
def test_signed_lag_from_known_attacks(calibration, lag):
    heads, y = rendered_attacks(lag)
    for size in FFT_SIZES:
        times, flux = attack_curve(y, size)
        result = lag_fit(times, flux, heads, calibration[str(size)]["correction_ms"])
        assert result["status"] == "resolved"
        assert abs(result["lag_ms"] - lag) <= 4.0


def test_mixed_time_blocks_are_unresolved(calibration):
    heads, y = rendered_attacks(np.repeat([-25.0, 25.0], 60))
    times, flux = attack_curve(y, FFT_SIZES[0])
    result = lag_fit(times, flux, heads, calibration[str(FFT_SIZES[0])]["correction_ms"])
    assert result["status"] == "unresolved"
    assert "unstable_time_blocks" in result["reasons"]


def test_silence_is_not_resolved(calibration):
    heads, y = rendered_attacks(0.0)
    times, flux = attack_curve(np.zeros_like(y), FFT_SIZES[0])
    result = lag_fit(times, flux, heads, calibration[str(FFT_SIZES[0])]["correction_ms"])
    assert result["status"] == "too_few_audible_anchors"


def test_chords_are_one_timing_observation():
    class Chart:
        notes = [
            ManiaNote(100, 0, NoteType.TAP),
            ManiaNote(100, 1, NoteType.TAP),
            ManiaNote(200, 0, NoteType.HOLD, 400),
        ]

    np.testing.assert_array_equal(head_times(Chart()), [100.0, 200.0])


def test_resume_rejects_changed_input_and_discards_partial_record(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    chart = tmp_path / "chart.osu"
    split = tmp_path / "split.json"
    audio.write_bytes(b"audio")
    chart.write_bytes(b"chart")
    split.write_text("{}")
    out = tmp_path / "results"
    task = {
        "audio": str(audio), "root": str(tmp_path), "charts": [chart.name], "group_id": "one",
    }
    monkeypatch.setattr(sys, "argv", [
        "analyze", "--root", str(tmp_path), "--split", str(split), "--output", str(out),
    ])
    monkeypatch.setattr(offsets, "calibrate", dict)
    monkeypatch.setattr(offsets, "select_groups", lambda *_: ([task], {}))

    class LocalPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, pending):
            for item in pending:
                yield {
                    "group_id": item["group_id"], "audio_sha256": offsets.digest(audio),
                    "results": [{
                        "relpath": chart.name, "sha256": offsets.digest(chart),
                        "status": "resolved",
                    }],
                }

    monkeypatch.setattr(offsets, "ProcessPoolExecutor", LocalPool)
    offsets.main()
    journal = out / "groups.jsonl"
    before = journal.read_bytes()
    journal.write_bytes(before + b'{"incomplete":')
    offsets.main()
    assert journal.read_bytes() == before
    assert len(json.loads(before)["results"]) == 1
    audio.write_bytes(b"other")
    with pytest.raises(ValueError, match="frozen audio input changed"):
        offsets.main()
    audio.write_bytes(b"audio")
    chart.write_bytes(b"other")
    with pytest.raises(ValueError, match="frozen chart input changed"):
        offsets.main()
