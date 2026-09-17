"""1/48 tick lattice. Train/infer window length is ``model.config.WINDOW_BARS``."""

TICKS_PER_BEAT = 48
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR
