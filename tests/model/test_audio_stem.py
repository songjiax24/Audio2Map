"""A span of stem rows matches the same frames from a longer span."""

from __future__ import annotations

import numpy as np

from audio2map.model.audio_stem import AudioStem, stem_span

# Overlapping frames are not bitwise equal. Each span builds its own mel
# buffer, so the filterbank and Conv1d see different lengths. float32 GEMM
# is not associative, and the kernel can depend on that shape. CI's largest
# gap was about 5.96e-8 (2**-24). This tolerance covers that gap.
_STEM_SPAN_ATOL = 1e-6


def test_segment_stem_matches_full_span() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(22_050 * 4).astype(np.float32) * 0.1
    stem = AudioStem()
    full = stem_span(y, 0.0, 4000.0, stem)
    left = stem_span(y, 0.0, 2000.0, stem)
    right = stem_span(y, 1500.0, 4000.0, stem)
    assert left.frame_lo == full.frame_lo
    for batch in (left, right):
        lo = batch.frame_lo - full.frame_lo
        hi = batch.frame_hi - full.frame_lo
        np.testing.assert_allclose(
            batch.features,
            full.features[lo:hi],
            rtol=0.0,
            atol=_STEM_SPAN_ATOL,
        )
