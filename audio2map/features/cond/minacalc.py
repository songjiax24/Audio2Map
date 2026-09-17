"""Etterna MinaCalc v515 in-process (SSR at score_goal 0.93)."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from audio2map.osu.schema import Beatmap

LIBRARY_ENV = "ETT_MINACALC_LIBRARY"
EXPECTED_VERSION = 515

_ETT_DIR = Path(__file__).resolve().parent / "ett"


@dataclass(frozen=True, slots=True)
class MsdScores:
    overall: float
    stream: float
    jumpstream: float
    handstream: float
    stamina: float
    jack_speed: float
    chordjack: float
    technical: float


class MinaCalcError(RuntimeError):
    pass


def _bundled_library() -> Path:
    if sys.platform == "win32":
        return _ETT_DIR / "minacalc.dll"
    if sys.platform == "darwin":
        return _ETT_DIR / "libminacalc.dylib"
    return _ETT_DIR / "libminacalc.so"


DEFAULT_LIBRARY = _bundled_library()

_lib: ctypes.CDLL | None = None
_lib_lock = threading.Lock()


def _resolve_library() -> Path:
    override = os.environ.get(LIBRARY_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{LIBRARY_ENV} points to missing file: {path}")
        return path
    if not DEFAULT_LIBRARY.is_file():
        raise FileNotFoundError(
            f"MinaCalc library not found at {DEFAULT_LIBRARY}. "
            f"Build it with audio2map/features/cond/ett/build.sh "
            f"or set {LIBRARY_ENV}."
        )
    return DEFAULT_LIBRARY


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = _resolve_library()
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise MinaCalcError(f"failed to load MinaCalc library {path}: {exc}") from exc
        lib.audio2map_minacalc_version.restype = ctypes.c_int
        lib.audio2map_minacalc_ssr.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.audio2map_minacalc_ssr.restype = ctypes.c_int
        version = int(lib.audio2map_minacalc_version())
        if version != EXPECTED_VERSION:
            raise MinaCalcError(
                f"MinaCalc version {version} at {path}, expected {EXPECTED_VERSION}"
            )
        _lib = lib
        return lib


def minacalc_version() -> int:
    return int(_load_lib().audio2map_minacalc_version())


def _note_rows(beatmap: Beatmap) -> list[tuple[int, float]]:
    """Build ``(lane_mask, time_sec)`` rows from note heads (Etterna convention)."""
    rows_by_time: dict[int, int] = {}
    for note in beatmap.notes:
        rows_by_time[note.time_ms] = rows_by_time.get(note.time_ms, 0) | (1 << note.col)
    return [(mask, t_ms / 1000.0) for t_ms, mask in sorted(rows_by_time.items())]


def compute_msd(
    beatmap: Beatmap,
    *,
    keys: int = 4,
    music_rate: float = 1.0,
    score_goal: float = 0.93,
) -> MsdScores:
    """SSR skillsets at ``score_goal`` (training labels use 0.93, not raw MSD mode)."""
    rows = _note_rows(beatmap)
    if len(rows) <= 1:
        return MsdScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    n = len(rows)
    masks = (ctypes.c_uint32 * n)(*(mask for mask, _ in rows))
    times = (ctypes.c_float * n)(*(t for _, t in rows))
    out = (ctypes.c_float * 8)()
    status = int(
        _load_lib().audio2map_minacalc_ssr(
            masks, times, n, music_rate, score_goal, keys, out
        )
    )
    if status != 0:
        raise MinaCalcError(f"MinaCalc SSR failed with status {status}")
    return MsdScores(*(float(out[i]) for i in range(8)))
