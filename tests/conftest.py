"""Shared pytest fixtures and markers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Librosa/numba need a writable cache; sandbox/CI often lack one next to site-packages.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".pytest_cache" / "numba"),
)

from audio2map.utils.paths import raw_data_available, raw_dir


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_data: needs AUDIO2MAP_DATA_ROOT with raw/ beatmaps (or specific fixtures)",
    )


@pytest.fixture(scope="session")
def data_raw_dir() -> Path:
    if not raw_data_available():
        pytest.skip("AUDIO2MAP_DATA_ROOT raw/ not present")
    return raw_dir()
