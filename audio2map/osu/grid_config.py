"""1/48 tick grid constants (Phase 1 v2)."""

TICKS_PER_BEAT = 48
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR  # 192

# Training / inference window length (fixed).
WINDOW_BARS = 16

POS_COUNT = TICKS_PER_BAR
