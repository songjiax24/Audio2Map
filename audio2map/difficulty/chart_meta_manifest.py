"""Load precomputed ``chart_meta/manifest.jsonl`` for training preload."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from audio2map.difficulty.chart_meta import ChartMeta
from audio2map.utils.paths import chart_meta_dir

logger = logging.getLogger(__name__)


def chart_meta_from_dict(data: dict) -> ChartMeta:
    fields = {f.name for f in ChartMeta.__dataclass_fields__.values()}
    return ChartMeta(**{k: v for k, v in data.items() if k in fields})


def load_chart_meta_manifest(path: Path | None = None) -> dict[str, ChartMeta]:
    """Return ``resolved_osu_path -> ChartMeta`` from manifest.jsonl."""
    manifest_path = path or (chart_meta_dir() / "manifest.jsonl")
    if not manifest_path.is_file():
        logger.warning("chart meta manifest missing: %s", manifest_path)
        return {}

    out: dict[str, ChartMeta] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                meta = chart_meta_from_dict(rec)
                out[str(Path(meta.osu_path).resolve())] = meta
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("skip manifest line %d: %s", line_no, exc)
    logger.info("loaded %d chart meta records from %s", len(out), manifest_path)
    return out
