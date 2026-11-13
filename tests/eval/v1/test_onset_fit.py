"""Negative controls must not score against the original audible event labels."""

import numpy as np
import torch

from audio2map.eval.v1.inputs import freeze_json
from audio2map.eval.v1.onset_fit import THRESHOLDS, evaluate
from audio2map.model.v1.onset import OnsetProbe


def test_silence_has_empty_reference_and_no_oracle_offset_measurement():
    probe = OnsetProbe(2)
    with torch.no_grad():
        probe.readout.weight.zero_()
        probe.readout.bias.copy_(torch.tensor([10.0, 0.0]))
    rows = [
        {
            "key": "silent",
            "kind": "review",
            "silent": True,
            "features": np.zeros((4, 2), dtype=np.float32),
            "frame_lo": 0,
            "present": np.ones(4, dtype=np.float32),
            "fraction": np.zeros(4),
            "onsets_ms": [0, 15, 30, 40],
        }
    ]
    score = evaluate(probe, rows, torch.zeros(2), torch.ones(2), torch.device("cpu"), 0.5)[0]
    assert score["scores"]["5"]["tp"] == score["scores"]["5"]["fn"] == 0
    assert score["scores"]["5"]["fp"] == 4
    assert score["false_positives_per_minute"] > 0
    assert score["oracle_presence_offset_mae_ms"] is None
    assert score["oracle_presence_offset_p95_ms"] is None


def test_fitting_thresholds_can_be_frozen_again(tmp_path):
    path = tmp_path / "fit_protocol.json"
    freeze_json(path, {"thresholds": THRESHOLDS})
    freeze_json(path, {"thresholds": THRESHOLDS})
