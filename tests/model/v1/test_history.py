"""A retained history keeps absolute RoPE positions and a fresh local ring."""

from types import SimpleNamespace

import torch

from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, DEC_SPAN
from audio2map.model.v1.kv import SelfKV, WindowStep


def test_graph_inputs_separate_absolute_position_from_ring_warmup():
    offset = 2 * DEC_SPAN
    kv = SelfKV(1, position_offset=offset)
    window = SimpleNamespace(rel_lo=0, rel_hi=CROSS_PAST + CROSS_FUTURE, keys=[torch.zeros(1)])
    stepper = WindowStep(None, kv, window, torch.zeros(1, 256))
    token = torch.ones(1, 1, dtype=torch.long)
    cursor = torch.ones(1, 1)
    assert not stepper.covers(0)
    stepper._load(token, 0, cursor)
    assert stepper.pos.item() == offset
    assert stepper.slot.item() == 0
    stepper._load(token, DEC_SPAN + 3, cursor)
    assert stepper.pos.item() == offset + DEC_SPAN + 3
    assert stepper.slot.item() == 3
    assert stepper.covers(DEC_SPAN + 3)
