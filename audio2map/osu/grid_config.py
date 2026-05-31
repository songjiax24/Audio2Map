"""1/48 tick grid constants (Phase 1 v2)."""

TICKS_PER_BEAT = 48
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR  # 192

# Training window sizes (debug starts with [8] only).
DEFAULT_WINDOW_BARS_CHOICES = (8,)

# Legacy names kept for docs referencing old segment design.
CONTEXT_BARS = 8
TARGET_BARS = 8
FUTURE_AUDIO_BARS = 2

POS_COUNT = TICKS_PER_BAR
