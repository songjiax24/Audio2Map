"""Skipped references and ROW-only hold endings need different explanations."""

import numpy as np

from audio2map.eval.v1.logit_stats import (
    reference_context,
    position_stats,
    row_stats,
)
from audio2map.eval.v1.logits import decision_distribution
from audio2map.osu.v1.export import write_v1_osu
from audio2map.osu.v1.timing import Redline
from audio2map.generate.v1.notes import V1Note, hit_lines_from_notes
from audio2map.tokens.v1.vocab import TOKEN_BEAT, build_vocab, pos_token


def reference(tmp_path):
    path = tmp_path / "reference.osu"
    write_v1_osu(
        path,
        hit_lines_from_notes([V1Note(100, 0, None), V1Note(200, 1, None)]),
        [],
        [Redline(0, 500, 4, 0)],
    )
    return reference_context(path, 1000)


def test_late_position_identifies_skipped_onset_with_an_available_correct_candidate(tmp_path):
    ref = reference(tmp_path)
    vocab = build_vocab()
    allowed = [vocab[pos_token(1, 5)], vocab[pos_token(2, 5)], vocab[TOKEN_BEAT]]
    logits = np.zeros(len(vocab), dtype=np.float32)
    logits[vocab[pos_token(2, 5)]] = 10
    row = {
        "beat_id": 0,
        "last_fraction": None,
        "chosen_token": pos_token(2, 5),
        "allowed_ids": allowed,
    }
    result = position_stats(row, decision_distribution(logits, allowed, 1), ref, set())
    assert result["skipped_onsets_ms"] == [100]
    assert result["target"]["token"] == pos_token(1, 5)
    assert result["target"]["legal"]
    assert not result["greedy_hits_next_reference"]
    assert result["chosen_has_reference"]


def test_previously_covered_nearby_reference_is_not_counted_as_a_new_skip(tmp_path):
    ref = reference(tmp_path)
    vocab = build_vocab()
    allowed = [vocab[pos_token(1, 5)], vocab[pos_token(2, 5)], vocab[TOKEN_BEAT]]
    logits = np.zeros(len(vocab), dtype=np.float32)
    row = {"beat_id": 0, "last_fraction": None, "chosen_token": TOKEN_BEAT, "allowed_ids": allowed}
    result = position_stats(row, decision_distribution(logits, allowed, 1), ref, {(0, 1, 5)})
    assert result["target"]["token"] == pos_token(2, 5)
    assert result["skipped_onsets_ms"] == [200]


def test_hold_ending_row_can_omit_a_reference_onset_without_a_position_error(tmp_path):
    ref = reference(tmp_path)
    vocab = build_vocab()
    allowed = [vocab["<ROW_4000>"], vocab["<ROW_4100>"]]
    logits = np.zeros(len(vocab), dtype=np.float32)
    row = {
        "beat_id": 0,
        "pending_fraction": [1, 5],
        "chosen_token": "<ROW_4000>",
        "allowed_ids": allowed,
    }
    result = row_stats(row, decision_distribution(logits, allowed, 1), ref)
    assert result["reference_has_onset"] and result["chosen_heads"] == 0
    assert result["head_count_probability"][0] == 0.5
