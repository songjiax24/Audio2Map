"""Per-type NLL uses the same buckets as split token accuracy."""

from __future__ import annotations

import math

import torch

from audio2map.metrics.nll import token_nll_by_type, token_nll_stats
from audio2map.metrics.token import loss_token_bucket
from audio2map.tokens import TOKEN_BAR, TOKEN_EOS, build_vocab


def test_type_nll_matches_token_cross_entropy() -> None:
    vocab = build_vocab()
    bar = vocab[TOKEN_BAR]
    eos = vocab[TOKEN_EOS]
    logits = torch.zeros(1, 2, len(vocab))
    logits[0, 0, bar] = 2.0
    logits[0, 1, eos] = 2.0
    targets = torch.tensor([[bar, eos]])
    mask = torch.ones(1, 2)
    buckets = token_nll_by_type(logits, targets, mask, id_to_token={i: t for t, i in vocab.items()})
    loss_sum, valid, _correct = token_nll_stats(logits, targets, mask)
    assert loss_token_bucket(TOKEN_BAR) == "bar"
    assert buckets["bar"][1] == 1
    assert buckets["eos"][1] == 1
    assert math.isclose(buckets["bar"][0] + buckets["eos"][0], float(loss_sum), rel_tol=1e-5)
    assert int(valid) == 2
