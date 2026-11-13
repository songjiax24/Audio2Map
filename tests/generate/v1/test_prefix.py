"""Forced prefixes use the production cursor/cache and leave sampling RNG untouched."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audio2map.generate.v1 import chart as charts
from audio2map.generate.v1 import sample as sampling
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.v1.codec import decode_tokens, encode_chart
from audio2map.tokens.v1.cursor_time import cursor_times_ms
from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKEN_BOS, build_vocab


def fixture():
    redlines = (Redline(0.0, 500.0, 4, 0),)
    encoded = encode_chart([ManiaNote(0, 0, NoteType.HOLD, 750)], redlines, 1000.0)
    return encoded, redlines


def setup(monkeypatch):
    encoded, _ = fixture()
    vocab = build_vocab()
    calls = []

    class Window:
        def __init__(self, *_args):
            pass

        def prepare(self, *_args):
            pass

    def logits(_model, token, position, kv, cond, c_t, **_kwargs):
        assert position == kv.length
        kv.commit()
        absolute = position + kv.position_offset
        calls.append((absolute, int(token.item()), float(c_t.item())))
        values = torch.zeros(1, 1, len(vocab))
        values[0, 0, vocab[encoded.tokens[absolute + 1]]] = 10
        return values

    monkeypatch.setattr(sampling, "CrossWindow", Window)
    monkeypatch.setattr(sampling, "cached_logits", logits)
    model = SimpleNamespace(training=False, decoder=[None])
    args = (model, torch.zeros(1, 2, 512), torch.arange(2).unsqueeze(0), encoded.grid)
    return encoded, args, calls


def test_open_hold_prefix_replays_positions_cursor_and_cache_without_random_draws(monkeypatch):
    encoded, args, calls = setup(monkeypatch)
    boundary = encoded.tokens.index(TOKEN_BEAT) + 1
    prefix = encoded.tokens[:boundary]
    draws = []
    choose = sampling._choose

    def record(logits, allowed, vocab, names, temperature, generator):
        draws.append(generator.get_state().clone())
        return choose(logits, allowed, vocab, names, temperature, generator)

    monkeypatch.setattr(sampling, "_choose", record)
    tokens = sampling.sample_chart(
        *args, cond=torch.zeros(1, 256), seed=7, temperature=0, prefix=prefix,
    )
    assert tokens == encoded.tokens
    assert len(draws) == len(tokens) - len(prefix)
    expected_rng = torch.Generator().manual_seed(7).get_state()
    assert torch.equal(draws[0], expected_rng)
    assert [call[0] for call in calls] == list(range(len(tokens) - 1))
    vocab = build_vocab()
    assert [call[1] for call in calls] == [vocab[token] for token in tokens[:-1]]
    expected_times = cursor_times_ms(tokens, encoded.grid)
    actual_times = [call[2] * sampling.DT_MS for call in calls]
    assert actual_times == pytest.approx(expected_times)
    events = decode_tokens(tokens, encoded.grid)
    assert [event.time_ms for event in events] == [0, 750]


def test_prefix_overrides_model_choice_and_complete_prefix_never_samples(monkeypatch):
    encoded, args, _ = setup(monkeypatch)
    monkeypatch.setattr(sampling, "_choose", lambda *_args: pytest.fail("unexpected draw"))
    tokens = sampling.sample_chart(
        *args, cond=torch.zeros(1, 256), seed=0, temperature=1, prefix=encoded.tokens,
    )
    assert tokens == encoded.tokens


@pytest.mark.parametrize("prefix", [[], [TOKEN_BEAT], [TOKEN_BOS, "<ROW_3000>"]])
def test_invalid_prefix_fails_before_decoder_work(monkeypatch, prefix):
    _encoded, args, calls = setup(monkeypatch)
    with pytest.raises(IllegalTokenError):
        sampling.sample_chart(
            *args, cond=torch.zeros(1, 256), seed=0, temperature=1, prefix=prefix,
        )
    assert calls == []


def test_bos_prefix_preserves_default_sampling(monkeypatch):
    _encoded, args, _calls = setup(monkeypatch)
    kwargs = {"cond": torch.zeros(1, 256), "seed": 3, "temperature": 0}
    assert sampling.sample_chart(*args, **kwargs) == sampling.sample_chart(
        *args, **kwargs, prefix=[TOKEN_BOS],
    )


def test_observer_preserves_tokens_and_reports_raw_logits_before_cursor_update(monkeypatch):
    encoded, args, _calls = setup(monkeypatch)
    prefix = encoded.tokens[:encoded.tokens.index(TOKEN_BEAT) + 1]
    kwargs = {"cond": torch.zeros(1, 256), "seed": 7, "temperature": 1, "prefix": prefix}
    expected = sampling.sample_chart(*args, **kwargs)
    records = []
    actual = sampling.sample_chart(*args, **kwargs, observer=records.append)
    assert actual == expected
    assert len(records) == len(actual) - 1
    times = cursor_times_ms(actual, encoded.grid)
    for index, record in enumerate(records):
        assert record.position == index
        assert record.input_token == actual[index]
        assert record.chosen_token == actual[index + 1]
        assert record.cursor_time_ms == pytest.approx(times[index])
        assert record.forced == (index + 1 < len(prefix))
        assert record.chosen_token in record.allowed_tokens
        assert record.raw_logits.device.type == "cpu"
        assert torch.isfinite(record.raw_logits).all()
    # Raw logits include tokens excluded by cursor legality.
    assert records[0].raw_logits[build_vocab()[TOKEN_BOS]] == 0


def test_observer_retains_absolute_positions_when_history_is_trimmed(monkeypatch):
    encoded, args, _calls = setup(monkeypatch)
    prefix = encoded.tokens[:encoded.tokens.index(TOKEN_BEAT) + 1]
    start = len(prefix) - 1
    records = []
    sampling.sample_chart(*args, cond=torch.zeros(1, 256), seed=0, temperature=0,
                          prefix=prefix, history_start=start, observer=records.append)
    assert records[0].position == start
    assert not records[0].forced


def test_short_history_keeps_cursor_and_open_hold_without_feeding_earlier_tokens(monkeypatch):
    encoded, args, calls = setup(monkeypatch)
    boundary = encoded.tokens.index(TOKEN_BEAT) + 1
    prefix = encoded.tokens[:boundary]
    start = boundary - 1
    tokens = sampling.sample_chart(
        *args, cond=torch.zeros(1, 256), seed=7, temperature=0,
        prefix=prefix, history_start=start,
    )
    assert tokens == encoded.tokens
    assert [call[0] for call in calls] == list(range(start, len(tokens) - 1))
    vocab = build_vocab()
    assert [call[1] for call in calls] == [vocab[token] for token in tokens[start:-1]]
    expected_times = cursor_times_ms(tokens, encoded.grid)[start:]
    assert [call[2] * sampling.DT_MS for call in calls] == pytest.approx(expected_times)
    assert [event.time_ms for event in decode_tokens(tokens, encoded.grid)] == [0, 750]


@pytest.mark.parametrize("start", [-1, 100, 1.5])
def test_invalid_history_start_fails_before_decoder_work(monkeypatch, start):
    encoded, args, calls = setup(monkeypatch)
    with pytest.raises(ValueError, match="history_start"):
        sampling.sample_chart(
            *args, cond=torch.zeros(1, 256), seed=0, temperature=0,
            prefix=encoded.tokens, history_start=start,
        )
    assert calls == []


@pytest.mark.parametrize("history_start", [0, 1])
def test_chart_generation_forwards_prefix_and_returns_history_with_continuation(
    monkeypatch, history_start,
):
    encoded, redlines = fixture()
    prefix = encoded.tokens[:encoded.tokens.index(TOKEN_BEAT) + 1]
    model = SimpleNamespace(eval=lambda: None)
    context = (encoded.grid, None, torch.zeros(1, 2, dtype=torch.long), 1)
    monkeypatch.setattr(charts, "_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(charts, "_condition", lambda *_args: None)

    def sample(*_args, **kwargs):
        assert kwargs["prefix"] == prefix
        assert kwargs["history_start"] == history_start
        return encoded.tokens

    monkeypatch.setattr(charts, "sample_chart", sample)
    result = charts.generate_v1_chart(
        model, np.zeros(1, dtype=np.float32), redlines, None,
        device=torch.device("cpu"), prefix=prefix, history_start=history_start,
    )
    assert result.tokens == tuple(encoded.tokens)
    assert result.notes[0].time_ms == 0
    assert result.notes[0].end_time_ms == 750
