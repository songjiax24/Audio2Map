"""Project-wide time tolerances, in milliseconds.

``EPS`` is the only coincidence tolerance. ``TIE`` is the fallback distance tie.
"""

from __future__ import annotations

# Red-line coincidence, window margin, and residual grid-point cutoff.
EPS = 1e-6

# Fallback nearest-point distances within this band are ties.
TIE = 1e-12
