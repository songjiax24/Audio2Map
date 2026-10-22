"""A time-0 note does not land on a grid point before the audio."""

from __future__ import annotations

import numpy as np

from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.labels import label_times


def test_time_zero_uses_the_legal_1_32_not_the_negative_downbeat() -> None:
    # Set 586296, first red line. Two beats earlier starts at about -0.31 ms.
    grid = build_beatgrid([Redline(784.0, 392.156862745098, 4, 0)], t_audio_ms=30_000.0)
    labels = label_times(grid, np.array([0.0]))
    assert int(labels.numer[0]) == 1
    assert int(labels.denom[0]) == 32
    assert labels.time_ms[0] > 0.0
    assert not labels.from_window[0]
