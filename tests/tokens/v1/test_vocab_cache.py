"""v1 vocab and event rows are built once."""

from __future__ import annotations

from audio2map.tokens.v1.vocab import build_vocab, event_rows, invert_vocab


def test_event_rows_and_vocab_are_process_caches() -> None:
    rows = event_rows()
    vocab = build_vocab()
    inverted = invert_vocab()
    assert event_rows() is rows
    assert build_vocab() is vocab
    assert invert_vocab() is inverted
    assert len(rows) == 609
    assert len(vocab) == 693
    assert inverted[vocab["<BEAT>"]] == "<BEAT>"
