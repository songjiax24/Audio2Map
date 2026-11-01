"""Mirror and clock-rate stay off unless training asks for them."""

from __future__ import annotations

import numpy as np

import pytest

from audio2map.dataset.v1.augment import (
    RATE_MAX,
    RATE_MIN,
    AugmentChoice,
    apply_clock,
    clock_chart,
    mirror_tokens,
    sample_augment,
)
from audio2map.tokens.v1.cursor_time import cursor_times_ms
from audio2map.dataset.v1.length import object_end_ms
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.audio.v1.stretch import stretch_pcm
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.train.v1.batches import epoch_batches


class _Draws:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)

    def uniform(self, low: float, high: float) -> float:
        drawn = self._values.pop(0)
        assert low == RATE_MIN
        assert high == RATE_MAX
        return drawn


def test_disabled_augment_is_rate_one() -> None:
    choice = sample_augment(np.random.default_rng(0), enabled=False)
    assert choice == AugmentChoice(mirror=False, rate=1.0)


def test_sample_augment_reads_mirror_then_rate() -> None:
    mirrored = sample_augment(_Draws([0.1, 0.9]), enabled=True)
    assert mirrored == AugmentChoice(mirror=True, rate=1.0)
    rated = sample_augment(_Draws([0.9, 0.1, 1.25]), enabled=True)
    assert rated == AugmentChoice(mirror=False, rate=1.25)


def test_clock_scales_redlines_and_pcm_and_keeps_note_times() -> None:
    notes = [
        ManiaNote(1000, 0, NoteType.TAP),
        ManiaNote(2000, 1, NoteType.HOLD, 2500),
    ]
    redlines = (Redline(0.0, 500.0, 4, 0),)
    pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
    scaled, lines, audio = apply_clock(notes, redlines, pcm, AugmentChoice(True, 2.0))
    assert scaled[0].time_ms == 1000
    assert scaled[0].col == 3
    assert scaled[1].col == 2
    assert scaled[1].end_time_ms == 2500
    assert lines[0].beat_length_ms == 250.0
    assert audio.shape[0] == SAMPLE_RATE // 2
    assert stretch_pcm(pcm, 1.0).shape == pcm.shape
    assert np.array_equal(stretch_pcm(pcm, 1.0), pcm)


def test_rate_keeps_tokens_and_scales_cursor_time() -> None:
    notes = [
        ManiaNote(100, 0, NoteType.TAP),
        ManiaNote(500, 1, NoteType.HOLD, 800),
    ]
    redlines = (Redline(0.0, 151.35, 4, 0),)
    pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
    base = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(False, 1.0))
    slow = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(False, 0.8))
    mirrored = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(True, 0.8))
    same_mirror = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(True, 1.0))
    assert slow.encoded.tokens == base.encoded.tokens
    assert "<POS_2/3>" in base.encoded.tokens
    assert mirrored.encoded.tokens == same_mirror.encoded.tokens
    assert mirrored.encoded.tokens != base.encoded.tokens
    base_times = cursor_times_ms(base.encoded.tokens, base.encoded.grid)
    slow_times = cursor_times_ms(slow.encoded.tokens, slow.encoded.grid)
    assert len(base_times) == len(slow_times)
    for original, stretched in zip(base_times, slow_times):
        assert stretched == pytest.approx(original / 0.8)
    phi, beat = base.encoded.grid.phase_at(80.0)
    scaled_phi, scaled_beat = slow.encoded.grid.phase_at(80.0 / 0.8)
    assert scaled_phi == pytest.approx(phi)
    assert scaled_beat == beat


def test_mirror_tokens_match_a_mirrored_chart() -> None:
    notes = [
        ManiaNote(100, 0, NoteType.TAP),
        ManiaNote(500, 1, NoteType.HOLD, 800),
    ]
    redlines = (Redline(0.0, 151.35, 4, 0),)
    pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
    plain = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(False, 1.0))
    mirrored = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(True, 1.0))
    assert mirror_tokens(plain.encoded.tokens) == mirrored.encoded.tokens


def test_cached_tokens_skip_labeling(monkeypatch: pytest.MonkeyPatch) -> None:
    notes = [
        ManiaNote(100, 0, NoteType.TAP),
        ManiaNote(500, 1, NoteType.HOLD, 800),
    ]
    redlines = (Redline(0.0, 151.35, 4, 0),)
    pcm = np.ones(SAMPLE_RATE, dtype=np.float32)
    plain = clock_chart(notes, redlines, pcm, 1000.0, AugmentChoice(False, 1.0))

    def cached(relpath: str, t_audio_ms: float) -> list[str]:
        assert relpath == "a.osu"
        assert t_audio_ms == 1000.0
        return list(plain.encoded.tokens)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("labeled a cached chart")

    monkeypatch.setattr("audio2map.dataset.v1.augment.cached_rate1_tokens", cached)
    monkeypatch.setattr("audio2map.dataset.v1.augment.encode_chart", forbidden)
    hit = clock_chart(
        notes, redlines, pcm, 1000.0, AugmentChoice(False, 1.25), relpath="a.osu"
    )
    assert hit.encoded.tokens == plain.encoded.tokens
    mirrored = clock_chart(
        notes, redlines, pcm, 1000.0, AugmentChoice(True, 1.0), relpath="a.osu"
    )
    assert mirrored.encoded.tokens == mirror_tokens(plain.encoded.tokens)


def test_object_end_includes_hold_tail(tmp_path) -> None:
    path = tmp_path / "end.osu"
    path.write_text(
        "\n".join(
            [
                "[HitObjects]",
                "64,192,1000,1,0,0:0:0:0:",
                "192,192,1500,128,0,1800:0:0:0:0:",
            ]
        ),
        encoding="utf-8",
    )
    assert object_end_ms(path) == 1800


def test_long_chart_is_a_batch_of_one() -> None:
    groups = epoch_batches(
        [10, 10, 100, 10, 10, 10],
        seed=0,
        epoch=0,
        budget=50,
        frac=0.1,
    )
    assert {tuple(sorted(group)) for group in groups} == {(0, 1, 3, 4, 5), (2,)}
