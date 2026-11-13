"""Mel frame step shared by the frontend and the beat grid.

``audio2map.grid`` cannot import ``audio2map.features``. Both import ``DT_MS``
from here. The step is two hops of the 22050 Hz, hop-128 lattice.
"""

from __future__ import annotations

SAMPLE_RATE = 22_050
HOP = 128
MS_PER_SECOND = 1_000.0
DT_MS = (2 * HOP) / SAMPLE_RATE * MS_PER_SECOND
