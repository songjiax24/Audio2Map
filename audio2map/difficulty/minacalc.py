"""Etterna MinaCalc MSD via bundled ``official_minacalc_runner`` binary."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from audio2map.osu.schema import Beatmap

RUNNER_ENV = "ETT_MINACALC_RUNNER"
DEFAULT_RUNNER = (
    Path(__file__).resolve().parent.parent / "third_party" / "ett" / "official_minacalc_runner"
)

SKILLSETS = (
    "Overall",
    "Stream",
    "Jumpstream",
    "Handstream",
    "Stamina",
    "JackSpeed",
    "Chordjack",
    "Technical",
)


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

    def as_dict(self) -> dict[str, float]:
        return {
            "Overall": self.overall,
            "Stream": self.stream,
            "Jumpstream": self.jumpstream,
            "Handstream": self.handstream,
            "Stamina": self.stamina,
            "JackSpeed": self.jack_speed,
            "Chordjack": self.chordjack,
            "Technical": self.technical,
        }


class MinaCalcError(RuntimeError):
    pass


def _resolve_runner() -> Path:
    override = os.environ.get(RUNNER_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{RUNNER_ENV} points to missing file: {path}")
        return path
    if not DEFAULT_RUNNER.is_file():
        raise FileNotFoundError(
            f"MinaCalc runner not found at {DEFAULT_RUNNER}. "
            f"Set {RUNNER_ENV} to a valid binary."
        )
    return DEFAULT_RUNNER


def _note_rows(beatmap: Beatmap, *, keys: int = 4) -> list[tuple[int, float]]:
    """Build ``(lane_mask, time_sec)`` rows from note heads (Etterna convention)."""
    lane_max = keys - 1
    rows_by_time: dict[int, int] = {}
    for note in beatmap.notes:
        lane = min(max(note.col, 0), lane_max)
        rows_by_time[note.time_ms] = rows_by_time.get(note.time_ms, 0) | (1 << lane)
    return [(mask, t_ms / 1000.0) for t_ms, mask in sorted(rows_by_time.items())]


def _build_payload(
    keys: int,
    music_rate: float,
    score_goal: float,
    rows: list[tuple[int, float]],
) -> str:
    lines = [f"{keys} {music_rate:.8f} {score_goal:.8f} {len(rows)}"]
    lines.extend(f"{mask} {t:.6f}" for mask, t in rows)
    return "\n".join(lines) + "\n"


def _parse_stdout(stdout: str) -> MsdScores:
    parts = stdout.strip().split()
    if len(parts) < 8:
        raise MinaCalcError(f"runner returned fewer than 8 values: {stdout.strip()!r}")
    try:
        nums = [float(v) for v in parts[:8]]
    except ValueError as exc:
        raise MinaCalcError(f"non-numeric runner output: {stdout.strip()!r}") from exc
    return MsdScores(*nums)


def compute_msd(
    beatmap: Beatmap,
    *,
    keys: int = 4,
    music_rate: float = 1.0,
    score_goal: float = 0.93,
    runner: Path | None = None,
) -> MsdScores:
    """Run MinaCalc on a parsed 4K beatmap."""
    rows = _note_rows(beatmap, keys=keys)
    if len(rows) <= 1:
        return MsdScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    runner_path = runner or _resolve_runner()
    payload = _build_payload(keys, music_rate, score_goal, rows)
    try:
        proc = subprocess.run(
            [str(runner_path)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except OSError as exc:
        raise MinaCalcError(f"failed to execute runner {runner_path}: {exc}") from exc

    if proc.returncode != 0:
        raise MinaCalcError(
            f"runner exit {proc.returncode}: stderr={proc.stderr.strip()!r} "
            f"stdout={proc.stdout.strip()!r}"
        )
    return _parse_stdout(proc.stdout)
