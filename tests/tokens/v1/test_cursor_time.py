"""Cursor time stays on the state that exists before the predicted token."""

from __future__ import annotations

from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import encode_chart
from audio2map.tokens.v1.cursor_time import cursor_times_ms, state_time_ms
from audio2map.tokens.v1.state import initial_cursor, step
from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKEN_BOS, TOKEN_EOS, parse_pos_token


def test_cursor_time_matches_the_state_before_each_target() -> None:
    notes = [
        ManiaNote(0, 0, NoteType.TAP),
        ManiaNote(250, 1, NoteType.TAP),
    ]
    encoded = encode_chart(notes, [Redline(0.0, 500.0, 4, 0)], 1000.0)
    tokens = encoded.tokens
    times = cursor_times_ms(tokens, encoded.grid)
    assert tokens[0] == TOKEN_BOS
    assert len(times) == len(tokens) - 1
    cursor = initial_cursor()
    saw_future_pos = False
    for index, token in enumerate(tokens[1:]):
        assert times[index] == state_time_ms(cursor, encoded.grid)
        if token.startswith("<POS_"):
            numer, denom = parse_pos_token(token)
            beat = encoded.grid.beats[cursor.beat_id]
            future = beat.start_ms + (numer / denom) * beat.nominal_length_ms
            if future != times[index]:
                saw_future_pos = True
                assert times[index] != future
        step(cursor, token, encoded.grid)
    assert saw_future_pos
    pos_half = tokens.index("<POS_1/2>")
    assert times[pos_half - 1] == 0.0
    row_after = pos_half + 1
    assert tokens[row_after].startswith("<ROW_")
    assert times[row_after - 1] == 250.0
    first_beat = tokens.index(TOKEN_BEAT)
    assert times[first_beat - 1] == 250.0
    assert times[first_beat] == 500.0
    assert tokens[-1] == TOKEN_EOS
    assert times[-1] == encoded.grid.coverage_end_ms


def test_hold_across_beats_uses_the_tail_only_after_its_pos() -> None:
    notes = [ManiaNote(0, 1, NoteType.HOLD, end_time_ms=750)]
    encoded = encode_chart(notes, [Redline(0.0, 500.0, 4, 0)], 1500.0)
    tokens = encoded.tokens
    times = cursor_times_ms(tokens, encoded.grid)
    tail = tokens.index("<POS_1/2>")
    assert times[tail - 1] == 500.0
    assert times[tail] == 750.0
    assert tokens[tail + 1].startswith("<ROW_")
