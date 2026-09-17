"""MinaCalc v515 shared library loads in-process."""

from __future__ import annotations

from pathlib import Path

from audio2map.features.cond.minacalc import (
    DEFAULT_LIBRARY,
    EXPECTED_VERSION,
    LIBRARY_ENV,
    compute_msd,
    minacalc_version,
)
from audio2map.osu.schema import Beatmap, ChartMetadata, ManiaNote, NoteType


def test_default_minacalc_library_exists() -> None:
    assert DEFAULT_LIBRARY.is_file(), (
        f"bundled MinaCalc library missing at {DEFAULT_LIBRARY}; "
        f"cond_vec MSD dims require it (rebuild with features/cond/ett/build.sh "
        f"or override via {LIBRARY_ENV})"
    )


def test_minacalc_version_is_515() -> None:
    assert minacalc_version() == EXPECTED_VERSION


def test_compute_msd_few_notes_non_negative() -> None:
    bm = Beatmap(
        path=Path("x.osu"),
        metadata=ChartMetadata("t", "a", "c", "v", None, None),
        notes=[
            ManiaNote(0, 0, NoteType.TAP),
            ManiaNote(100, 1, NoteType.TAP),
            ManiaNote(200, 2, NoteType.TAP),
            ManiaNote(300, 3, NoteType.TAP),
        ],
    )
    msd = compute_msd(bm)
    assert msd.overall >= 0.0
    assert msd.stream >= 0.0
