"""Data structures for parsed osu! beatmaps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NoteType(Enum):
    TAP = "tap"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class ManiaNote:
    """Single mania key event at ``time_ms`` in column ``col`` (0-3 for 4K)."""

    time_ms: int
    col: int
    note_type: NoteType
    end_time_ms: int | None = None  # hold tail time; None for taps

    def __post_init__(self) -> None:
        if self.col < 0 or self.col > 3:
            raise ValueError(f"4K column must be 0-3, got {self.col}")
        if self.note_type == NoteType.HOLD:
            if self.end_time_ms is None:
                raise ValueError("hold notes require end_time_ms")
            if self.end_time_ms <= self.time_ms:
                raise ValueError(
                    f"hold end ({self.end_time_ms}) must be after start ({self.time_ms})"
                )
        elif self.end_time_ms is not None:
            raise ValueError("tap notes must not have end_time_ms")


@dataclass(frozen=True, slots=True)
class TimingPoint:
    offset_ms: int
    beat_length_ms: float
    meter: int
    uninherited: bool


@dataclass(frozen=True, slots=True)
class ChartMetadata:
    title: str
    artist: str
    creator: str
    version: str
    beatmap_id: int | None
    beatmap_set_id: int | None


@dataclass(slots=True)
class Beatmap:
    path: Path
    metadata: ChartMetadata
    timing_points: list[TimingPoint] = field(default_factory=list)
    notes: list[ManiaNote] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.notes)
