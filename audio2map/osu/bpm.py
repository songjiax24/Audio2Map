"""BPM canonicalization for Phase 1 tick grid."""

from __future__ import annotations


def canonicalize_bpm(original_bpm: float) -> tuple[float, int]:
    """Map BPM to ``[120, 240)`` via powers of two. Returns ``(canonical_bpm, scale_exp)``."""
    if original_bpm <= 0:
        raise ValueError(f"invalid BPM: {original_bpm}")
    bpm = float(original_bpm)
    scale_exp = 0
    while bpm < 120.0:
        bpm *= 2.0
        scale_exp += 1
    while bpm >= 240.0:
        bpm /= 2.0
        scale_exp -= 1
    return bpm, scale_exp
