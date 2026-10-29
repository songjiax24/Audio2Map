"""Frame and token counts that match a whole-chart forward.

``n_frames`` is ``frame_index(coverage_end) - frame_index(coverage_start)``
on the beat grid of this audio duration. ``n_tokens`` is ``len(encode_chart)``,
the same length as ``token_ids``. Neither number is the object-end time.
"""

from __future__ import annotations

from audio2map.features.audio.v1.mel import frame_index
from audio2map.osu.schema import ManiaNote
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import encode_chart


def chart_shape(
    notes: list[ManiaNote],
    redlines: tuple[Redline, ...] | list[Redline],
    t_audio_ms: float,
) -> tuple[int, int]:
    """Return ``(n_frames, n_tokens)`` for one chart at this audio duration."""
    encoded = encode_chart(list(notes), redlines, t_audio_ms)
    grid = encoded.grid
    n_frames = frame_index(grid.coverage_end_ms) - frame_index(grid.coverage_start_ms)
    return n_frames, len(encoded.tokens)
