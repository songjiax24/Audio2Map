"""Chart-level failures. One reason constant, one situation."""

from __future__ import annotations

NO_REDLINE = "no_redline"
INVALID_TIMING_FIELD = "invalid_timing_field"
NON_FINITE_OFFSET = "non_finite_offset"
NONPOSITIVE_BEAT_LENGTH = "nonpositive_beat_length"
DUPLICATE_REDLINE = "duplicate_redline"
BEAT_LENGTH_BELOW_MINIMUM = "beat_length_below_minimum"
TOO_MANY_BEATS = "too_many_beats"
BAD_AUDIO_DURATION = "bad_audio_duration"
NO_BEAT_BEFORE_ZERO = "no_beat_before_zero"
GRID_SHORT_OF_AUDIO = "grid_short_of_audio"
BEAT_INDEX_NOT_STARTED = "beat_index_not_started"
RESIDUAL_WITHOUT_REDLINE = "residual_without_redline"
ENDPOINT_OUTSIDE_AUDIO = "endpoint_outside_audio"
DUPLICATE_ENDPOINT = "duplicate_endpoint"
HOLD_NOT_AFTER_HEAD = "hold_not_after_head"
NO_LEGAL_GRID = "no_legal_grid"
OPEN_HOLD_AT_END = "open_hold_at_end"


class ChartRejected(ValueError):
    """The whole chart is unusable. ``reason`` is the corpus-stat key."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)
