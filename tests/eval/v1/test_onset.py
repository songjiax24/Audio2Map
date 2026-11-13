"""Physical onset bins and matching must not hide subframe errors or collisions."""

import numpy as np
import pytest
import torch
from torch import nn

from audio2map.eval.v1 import onset
from audio2map.eval.v1.onset import capture_features, frame_targets, onset_match, predicted_onsets
from audio2map.features.audio.v1.constants import DT_MS
from audio2map.osu.v1.timing import Redline


def test_targets_round_trip_subframe_times_and_negative_frame_origin():
    times = np.array([0, 17.3, 40.5, 81.0])
    target = frame_targets(times, -2, 20)
    decoded = predicted_onsets(target.present, target.fraction, -2, 0.5)
    assert decoded == pytest.approx(times, abs=1e-5)
    assert target.present.sum() == len(times)


@pytest.mark.parametrize("times", [[1, 2], [5, 5], [-100], [10000], [float("nan")]])
def test_invalid_or_unrepresentable_targets_are_rejected(times):
    with pytest.raises(ValueError):
        frame_targets(times, 0, 10)


def test_match_is_inclusive_one_to_one_and_does_not_round_fractional_times():
    assert onset_match([5, 0], [2, 7], 2)["tp"] == 2
    assert onset_match([0, 1], [0], 2)["fp"] == 1
    assert onset_match([0.1], [0], 0)["tp"] == 0
    assert onset_match([], [], 2)["f1"] == 0


def test_frame_width_is_not_a_matching_tolerance():
    target = frame_targets([DT_MS / 2], 0, 2)
    decoded = predicted_onsets(target.present, np.zeros(2), 0, 0.5)
    assert onset_match(decoded, [DT_MS / 2], 2)["tp"] == 0


def test_sorted_match_agrees_with_maximum_cardinality_dynamic_programming():
    rng = np.random.default_rng(7)
    for _ in range(100):
        pred = np.sort(rng.integers(0, 20, size=6)).tolist()
        ref = np.sort(rng.integers(0, 20, size=5)).tolist()
        table = np.zeros((len(pred) + 1, len(ref) + 1), dtype=int)
        for i, p in enumerate(pred, 1):
            for j, r in enumerate(ref, 1):
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
                if abs(p - r) <= 2:
                    table[i, j] = max(table[i, j], table[i - 1, j - 1] + 1)
        assert onset_match(pred, ref, 2)["tp"] == table[-1, -1]


def fake_model():
    class Stem(nn.Module):
        def forward(self, mel):
            return mel[:, :, 2:-1:2].transpose(1, 2)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = Stem()

        def forward(self, mel):
            return self.stem(mel) * 2

    model = nn.Module()
    model.encoder = Encoder()
    model.phase = nn.Identity()
    return model.eval()


def test_capture_uses_exact_phase_output_and_removes_hooks(monkeypatch):
    model = fake_model()
    expected = {}

    def native_encode(model, _pcm, _grid, _redlines, **_kwargs):
        mel = torch.arange(22, dtype=torch.float32).reshape(1, 2, 11)
        memory = model.encoder(mel)
        phase = model.phase(torch.full_like(memory, 0.125))
        expected.update(memory=memory, phase=phase, context=memory + phase)
        return memory + phase, torch.arange(-2, 2).unsqueeze(0), 1

    monkeypatch.setattr(onset, "encode_context", native_encode)
    features, frame_lo = capture_features(
        model,
        np.zeros(22050, dtype=np.float32),
        [Redline(0, 500, 4, 0)],
        device=torch.device("cpu"),
    )
    assert frame_lo == -2
    assert features["mel"].shape == (4, 2)
    np.testing.assert_array_equal(features["encoder"], expected["memory"][0])
    np.testing.assert_array_equal(features["phase"], expected["phase"][0])
    np.testing.assert_array_equal(features["context"], expected["context"][0])
    assert all(not module._forward_hooks for module in model.modules())


def test_capture_cleans_up_hooks_after_an_encoder_failure(monkeypatch):
    model = fake_model()

    def fail(*_args, **_kwargs):
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(onset, "encode_context", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        capture_features(
            model, np.zeros(22050), [Redline(0, 500, 4, 0)], device=torch.device("cpu")
        )
    assert all(not module._forward_hooks for module in model.modules())
