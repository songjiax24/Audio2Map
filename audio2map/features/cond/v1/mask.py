"""Composer-style condition mask. v1 does not train CFG."""

from __future__ import annotations

import numpy as np

KEEP_ALL_P = 0.10
DROP_ALL_P = 0.10
P_KEEP = 0.5


def sample_cond_mask(m_real: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw m, then ``m = m AND m_real``. A missing dim cannot be turned on.

    With probability 0.10 every present dim is kept. With probability 0.10
    every dim is dropped. Otherwise each dim is an independent Bernoulli
    with ``p_keep = 0.5``. Pattern dims are ordinary dims: dropping one does
    not renorm the others.
    """
    real = np.asarray(m_real, dtype=np.float32)
    draw = float(rng.random())
    if draw < KEEP_ALL_P:
        chosen = np.ones(real.shape, dtype=np.float32)
    elif draw < KEEP_ALL_P + DROP_ALL_P:
        chosen = np.zeros(real.shape, dtype=np.float32)
    else:
        chosen = (rng.random(real.shape) < P_KEEP).astype(np.float32)
    return chosen * real
