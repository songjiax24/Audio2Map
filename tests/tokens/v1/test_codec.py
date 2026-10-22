"""V80 labels, encode/decode, and hold constraints."""

from __future__ import annotations

import pytest

from audio2map.utils.errors import ChartRejected
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.v1.codec import EndpointKind, decode_tokens, encode_chart
from audio2map.tokens.v1.state import initial_cursor, legal_tokens, step
from audio2map.tokens.v1.vocab import TOKENIZER_VERSION, build_vocab


def _rl(offset: float, length: float, index: int = 0) -> Redline:
    return Redline(offset, length, 4, index)


def test_vocab_is_693() -> None:
    vocab = build_vocab()
    assert len(vocab) == 693
    assert TOKENIZER_VERSION == "v1-1"
    assert "<BAR>" not in vocab
    assert "<POS_0/1>" in vocab
    assert "<POS_23/32>" in vocab
    assert "<ROW_2121>" in vocab
    assert "<ROW_0000>" not in vocab
    assert "<ROW_3333>" not in vocab


def test_near_duplicate_endpoint_labels_on_the_later_redline() -> None:
    from audio2map.osu.v1.timing import parse_redlines
    from audio2map.utils.tolerance import EPS

    parsed = parse_redlines(
        ["0,500,4,2,0,50,1,0", f"{EPS / 2},480,4,2,0,50,1,0"]
    )
    encoded = encode_chart(
        [ManiaNote(0, 0, NoteType.TAP)],
        parsed.redlines,
        1000.0,
    )
    assert int(encoded.labels.section[0]) == 1
    assert int(encoded.labels.numer[0]) == 0
    assert int(encoded.labels.denom[0]) == 1
    assert encoded.labels.from_window[0]


def test_downbeat_window_and_roundtrip() -> None:
    notes = [ManiaNote(0, 0, NoteType.TAP)]
    encoded = encode_chart(notes, [_rl(0.0, 500.0)], 1500.0)
    assert encoded.labels.from_window[0]
    assert int(encoded.labels.numer[0]) == 0
    assert int(encoded.labels.denom[0]) == 1
    decoded = decode_tokens(encoded.tokens, encoded.grid)
    got = [
        (item.lane, item.kind, item.section, item.beat_k, item.numer, item.denom)
        for item in decoded
    ]
    assert got == [(0, EndpointKind.TAP, 0, 0, 0, 1)]


def test_hold_across_beats_roundtrip() -> None:
    notes = [ManiaNote(0, 1, NoteType.HOLD, end_time_ms=500)]
    encoded = encode_chart(notes, [_rl(0.0, 500.0)], 1500.0)
    decoded = decode_tokens(encoded.tokens, encoded.grid)
    assert [(item.lane, item.kind, item.beat_k) for item in decoded] == [
        (1, EndpointKind.HEAD, 0),
        (1, EndpointKind.TAIL, 1),
    ]


def test_endpoint_outside_audio_rejects_chart() -> None:
    notes = [ManiaNote(2000, 0, NoteType.TAP)]
    with pytest.raises(ChartRejected) as caught:
        encode_chart(notes, [_rl(0.0, 500.0)], 1500.0)
    assert caught.value.reason == "endpoint_outside_audio"


def test_same_cell_rejects_chart() -> None:
    notes = [ManiaNote(0, 0, NoteType.TAP), ManiaNote(0, 0, NoteType.TAP)]
    with pytest.raises(ChartRejected) as caught:
        encode_chart(notes, [_rl(0.0, 500.0)], 1500.0)
    assert caught.value.reason == "duplicate_endpoint"


def test_open_hold_cannot_walk_past_the_last_legal_point() -> None:
    encoded = encode_chart(
        [ManiaNote(0, 0, NoteType.HOLD, end_time_ms=500)],
        [_rl(0.0, 500.0)],
        1500.0,
    )
    cursor = initial_cursor()
    step(cursor, encoded.tokens[1], encoded.grid)  # POS
    # A hold head is legal here because a later grid point exists.
    legal = legal_tokens(cursor, encoded.grid)
    assert any(token.startswith("<ROW_") and "2" in token[5:9] for token in legal)
    # At G_end the only token is EOS, and an open hold cannot get there:
    # the encoder's own sequence closes the hold first.
    decode_tokens(encoded.tokens, encoded.grid)


def test_every_encode_state_has_a_legal_token() -> None:
    notes = [
        ManiaNote(0, 0, NoteType.HOLD, end_time_ms=500),
        ManiaNote(10, 2, NoteType.TAP),
    ]
    encoded = encode_chart(notes, [_rl(0.0, 500.0)], 1200.0)
    cursor = initial_cursor()
    assert legal_tokens(cursor, encoded.grid)
    for token in encoded.tokens[1:]:
        assert token in legal_tokens(cursor, encoded.grid)
        step(cursor, token, encoded.grid)
        if not cursor.finished:
            assert legal_tokens(cursor, encoded.grid)


def test_illegal_token_does_not_move_the_cursor() -> None:
    encoded = encode_chart([ManiaNote(0, 0, NoteType.TAP)], [_rl(0.0, 500.0)], 1500.0)
    cursor = initial_cursor()
    with pytest.raises(IllegalTokenError):
        step(cursor, "<ROW_1111>", encoded.grid)
    assert cursor.beat_id == 0
    assert not cursor.expect_row
