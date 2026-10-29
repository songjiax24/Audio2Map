"""Clock-rate stretch of PCM that is already at the model sample rate.

The clock runs at ``rate``: duration becomes ``1/rate`` and pitch moves with
it. This is the same soxr HQ resample as the decode path, not a phase
vocoder and not a separate pitch shift. ``rate=1`` returns the buffer.
"""

from __future__ import annotations

import math

import numpy as np

from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.audio.v1.decode import resample_hq


def stretch_pcm(pcm: np.ndarray, rate: float) -> np.ndarray:
    """Return PCM at ``SAMPLE_RATE`` whose timeline is the original divided by ``rate``."""
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate must be positive and finite, got {rate}")
    audio = np.asarray(pcm, dtype=np.float32)
    if rate == 1.0:
        return audio
    return resample_hq(audio, SAMPLE_RATE * rate, SAMPLE_RATE)
