"""Object-end time of a chart file.

Training batches do not use this. The v1 loop buckets on mel frames and
token counts (``dataset/v1/shape.py``). This helper remains for inspections
that only need the last hit-object time.
"""

from __future__ import annotations

from pathlib import Path


def object_end_ms(path: Path | str) -> int:
    """Last hit-object time in the file, including a hold tail. Empty is 0."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = lines.index("[HitObjects]") + 1
    except ValueError:
        return 0
    end = 0
    for line in lines[start:]:
        if line.startswith("["):
            break
        parts = line.split(",")
        if len(parts) < 3 or not parts[2].strip():
            continue
        end = max(end, int(float(parts[2])))
        extras = parts[5] if len(parts) > 5 else ""
        if ":" in extras:
            end = max(end, int(float(extras.split(":", 1)[0])))
    return end
