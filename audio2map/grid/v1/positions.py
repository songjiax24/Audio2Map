"""V80 position cells. Shared by the grid and the v1 tokenizer."""

from __future__ import annotations

import math

# Factor-closed A' denominators. 18 is included.
V80_DENS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 24, 32)


def v80_cells() -> tuple[tuple[int, int], ...]:
    """Reduced ``(numerator, denominator)`` sorted by ``(d, k)``. Length 80."""
    cells = [
        (numer, denom)
        for denom in V80_DENS
        for numer in range(denom)
        if math.gcd(numer, denom) == 1
    ]
    cells.sort(key=lambda item: (item[1], item[0]))
    if len(cells) != 80:
        raise RuntimeError(f"V80 has {len(cells)} cells")
    return tuple(cells)
