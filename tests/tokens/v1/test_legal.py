"""Gold targets stay inside the legal ids, and the tally reads those rows."""

from __future__ import annotations

import torch

from audio2map.model.v1.loss import legal_tally
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.state import (
    _token_allowed,
    initial_cursor,
    legal_rows,
    legal_tokens,
    step,
)
from audio2map.tokens.v1.vocab import build_vocab, invert_vocab


def _encoded():
    notes = [
        ManiaNote(0, 0, NoteType.TAP),
        ManiaNote(500, 1, NoteType.HOLD, end_time_ms=1000),
        ManiaNote(1500, 2, NoteType.TAP),
    ]
    return encode_chart(notes, (Redline(0.0, 500.0, 4, 0),), 2500.0)


def test_one_token_check_matches_the_legal_set() -> None:
    encoded = _encoded()
    cursor = initial_cursor()
    probes = {"<PAD>", "<BOS>", "<EOS>", "<BEAT>", "<POS_1/2>", "<ROW_0000>"}
    for gold in encoded.tokens[1:]:
        allowed = legal_tokens(cursor, encoded.grid)
        for token in allowed | probes:
            assert _token_allowed(cursor, token, encoded.grid) == (token in allowed)
        step(cursor, gold, encoded.grid)


def test_legal_rows_match_the_cursor() -> None:
    encoded = _encoded()
    vocab = build_vocab()
    names = invert_vocab(vocab)
    ids = [vocab[token] for token in encoded.tokens]
    rows = legal_rows(ids, encoded.grid)
    cursor = initial_cursor()
    for index, gold in enumerate(encoded.tokens[1:]):
        allowed = legal_tokens(cursor, encoded.grid)
        got = {names[int(token_id)] for token_id in rows[index]}
        assert got == allowed
        assert gold in allowed
        step(cursor, gold, encoded.grid, allowed=allowed)


def test_tally_is_one_when_the_argmax_is_gold() -> None:
    encoded = _encoded()
    vocab = build_vocab()
    ids = [vocab[token] for token in encoded.tokens]
    rows = legal_rows(ids, encoded.grid)
    targets = torch.tensor([ids[1:]], dtype=torch.long)
    logits = torch.full((1, targets.shape[1], len(vocab)), -20.0)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    tally = legal_tally(logits, targets, rows)
    assert tally.count == len(rows)
    assert tally.hits == tally.count
    assert abs(tally.mass - tally.count) < 1e-4
    assert sum(item[2] for item in tally.by_type.values()) == tally.count
    for name in ("pos", "row", "beat", "eos"):
        type_hits, type_mass, type_count = tally.by_type[name]
        assert type_count > 0
        assert type_hits == type_count
        assert abs(type_mass - type_count) < 1e-4


def test_tally_drops_when_the_argmax_is_outside_the_row() -> None:
    encoded = _encoded()
    vocab = build_vocab()
    ids = [vocab[token] for token in encoded.tokens]
    rows = legal_rows(ids, encoded.grid)
    targets = torch.tensor([ids[1:]], dtype=torch.long)
    illegal = vocab["<PAD>"]
    assert illegal not in set(int(token_id) for token_id in rows[0])
    logits = torch.full((1, targets.shape[1], len(vocab)), -20.0)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    logits[:, 0, :] = -20.0
    logits[:, 0, illegal] = 20.0
    tally = legal_tally(logits, targets, rows)
    assert tally.hits == tally.count - 1
    assert tally.mass < tally.count - 0.5
    kind = encoded.tokens[1].split("_", 1)[0].strip("<>").lower()
    dropped_hits, _, dropped_count = tally.by_type[kind]
    assert dropped_hits == dropped_count - 1
    for name, (type_hits, _, type_count) in tally.by_type.items():
        if name != kind:
            assert type_hits == type_count
