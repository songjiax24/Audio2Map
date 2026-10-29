"""Composer-style condition mask. v1 does not train CFG."""

from __future__ import annotations

import hashlib

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


def frozen_partial_mask(m_real: np.ndarray, *, seed: int, key: str) -> np.ndarray:
    """Partial mask for one chart. The same ``seed`` and ``key`` redraw it.

    Each present dim is an independent Bernoulli with ``p_keep``. Missing dims
    stay off. This is the middle training branch: it does not take the keep-all
    or drop-all draw. With two or more present dims, a draw that keeps all of
    them or none of them is discarded.
    """
    digest = hashlib.sha256(f"{seed}\n{key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    real = np.asarray(m_real, dtype=np.float32).reshape(-1)
    n_present = int((real > 0.0).sum())
    chosen = np.zeros(real.shape, dtype=np.float32)
    for _ in range(64):
        chosen = (rng.random(real.shape) < P_KEEP).astype(np.float32) * real
        if n_present < 2:
            return chosen
        kept = int((chosen > 0.0).sum())
        if 0 < kept < n_present:
            return chosen
    return chosen
