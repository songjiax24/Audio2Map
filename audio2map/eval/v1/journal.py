"""Durable progress for long eval runs; reuse requires the frozen run inputs to match."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio2map.eval.v1.inputs import matches_digest
from audio2map.utils.time import SAMPLE_RATE


def write_playback_pcm(path: Path, pcm: np.ndarray) -> None:
    """Persist the model input without resampling or quantizing it for playback."""
    temporary = path.with_suffix(".part")
    sf.write(temporary, pcm, SAMPLE_RATE, format="WAV", subtype="FLOAT")
    temporary.replace(path)


def record_key(row: dict) -> tuple:
    return row["relpath"], row["scenario"], row["mode"], row.get("seed")


def read_records(path: Path, *, key=record_key) -> tuple[dict, int]:
    """Read latest committed records without modifying the journal."""
    previous = {}
    valid_bytes = 0
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row = json.loads(line)
        previous[key(row)] = row
        valid_bytes += len(line)
    return previous, valid_bytes


class Journal:
    def __init__(self, path: Path, *, resume: bool):
        self.path = path
        self.rows: list[dict] = []
        self.previous = {}
        if resume and path.is_file():
            self.previous, valid_bytes = read_records(path)
            # Only an uncommitted final line is discarded; corrupt complete lines fail.
            if valid_bytes != path.stat().st_size:
                with path.open("r+b") as stream:
                    stream.truncate(valid_bytes)
        self.reused = 0
        self.stream = path.open("a" if resume else "w", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def get(self, base: dict, mode: str, seed: int | None = None) -> dict | None:
        row = self.previous.get(record_key({**base, "mode": mode, "seed": seed}))
        if row is not None and all(row.get(name) == value for name, value in base.items()):
            return row
        return None

    def add(self, row: dict) -> None:
        if self.previous.get(record_key(row)) is row:
            self.reused += 1
        else:
            json.dump(row, self.stream, ensure_ascii=False, allow_nan=False)
            self.stream.write("\n")
            self.stream.flush()
            self.previous[record_key(row)] = row
        self.rows.append(row)


def generated_tokens(row: dict | None) -> list[str] | None:
    if row is None:
        return None
    output = Path(row["output"])
    tokens = output.with_name("tokens.json")
    if not matches_digest(output, row.get("osu_sha256")):
        return None
    if not matches_digest(tokens, row.get("tokens_sha256")):
        return None
    if row.get("shift_samples") and (
        not row.get("playback_audio")
        or not matches_digest(output.with_name(row["playback_audio"]), row.get("playback_sha256"))
    ):
        return None
    return json.loads(tokens.read_text())
