import json

import pytest

from audio2map.eval.v1.protocol import load_protocol, pattern_levels


def test_pattern_levels_preserve_zero_mass_and_calibrate_positive_range():
    result = pattern_levels([0, 0, 0.02, 0.04, None])
    assert result["distribution"]["count"] == 4
    assert result["zero_count"] == result["positive_count"] == 2
    assert result["levels"] == pytest.approx([0, 0.025, 0.035])
    assert pattern_levels([0, 0])["levels"] == []


def test_protocol_validates_baseline_order_and_pattern_bounds(tmp_path):
    path = tmp_path / "protocol.json"
    valid = {"protocol_version": 1, "audio_offsets_s": [-0.1, 0, 0.1],
             "controls": {"pattern_jacks": [0, 0.1, 0.3]}}
    path.write_text(json.dumps(valid))
    protocol = load_protocol(path)
    assert protocol.offsets_s == (-0.1, 0.0, 0.1)
    assert protocol.controls["pattern_jacks"] == (0, 0.1, 0.3)
    for offsets, levels in (([-0.1, 0.1], [0, 0.3]), ([0], [0.3, 0.1]), ([0], [0, 1.1])):
        path.write_text(json.dumps({**valid, "audio_offsets_s": offsets,
                                    "controls": {"pattern_jacks": levels}}))
        with pytest.raises(ValueError):
            load_protocol(path)
