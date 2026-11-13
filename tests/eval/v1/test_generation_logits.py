"""Raw/grammar/policy distributions and recorder coordinates remain distinct."""

import numpy as np
import pytest
import torch

from audio2map.eval.v1.logits import decision_distribution, token_stats
from audio2map.generate.v1.sample import sampling_distribution
from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKEN_BOS, build_vocab, pos_token


def test_legal_rank_and_raw_rank_do_not_confuse_grammar_with_preference():
    vocab = build_vocab()
    correct = pos_token(0, 1)
    logits = np.zeros(len(vocab), dtype=np.float32)
    logits[vocab[TOKEN_BOS]] = 20
    logits[vocab[correct]] = 2
    logits[vocab[TOKEN_BEAT]] = 1
    allowed = [vocab[correct], vocab[TOKEN_BEAT]]
    distribution = decision_distribution(logits, allowed, 1.0)
    stat = token_stats(distribution, correct, allowed)
    assert stat["raw_rank"] == 2 and stat["legal_rank"] == 1
    assert stat["legal_probability"] == pytest.approx(1 / (1 + np.exp(-1)))
    assert stat["raw_probability"] < 0.001
    assert distribution["legal_mass_raw"] < 0.001


def test_temperature_changes_probabilities_and_greedy_has_no_illegal_mass():
    vocab = build_vocab()
    allowed = {pos_token(0, 1), TOKEN_BEAT}
    logits = torch.zeros(len(vocab))
    logits[vocab[pos_token(0, 1)]] = 2
    logits[vocab[TOKEN_BEAT]] = 1
    p = sampling_distribution(logits, allowed, vocab, 0.5)
    assert p[vocab[pos_token(0, 1)]] == pytest.approx(1 / (1 + np.exp(-2)))
    greedy = sampling_distribution(logits, allowed, vocab, 0)
    assert greedy.sum() == 1 and greedy[vocab[pos_token(0, 1)]] == 1
    assert p[vocab[TOKEN_BOS]] == 0
