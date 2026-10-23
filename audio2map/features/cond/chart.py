"""Per-chart condition table (SR, Etterna MSD, pattern stats) that feeds ``cond_vec``."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from audio2map.features.cond.minacalc import compute_msd
from audio2map.features.cond.official_sr import official_star_rating
from audio2map.grid import CanonicalTiming
from audio2map.osu.parser import parse_beatmap
from audio2map.utils.paths import chart_meta_dir

logger = logging.getLogger(__name__)


class ChartMetaVersionError(ValueError):
    """A saved chart-meta record is not the current v1 condition vector."""


def _v1_cond_version() -> int:
    from audio2map.features.cond.v1.vec import COND_VEC_VERSION

    return COND_VEC_VERSION


@dataclass(slots=True)
class ChartMeta:
    osu_path: str
    beatmap_id: int | None = None
    canonical_bpm: float | None = None
    official_sr: float | None = None
    rework_sr: float | None = None
    msd_overall: float | None = None
    msd_stream: float | None = None
    msd_jumpstream: float | None = None
    msd_handstream: float | None = None
    msd_stamina: float | None = None
    msd_jack_speed: float | None = None
    msd_chordjack: float | None = None
    msd_technical: float | None = None
    analyzer_ln_percent: float | None = None
    analyzer_hb_row_ratio: float | None = None
    analyzer_stream: float | None = None
    analyzer_chordstream: float | None = None
    analyzer_jacks: float | None = None
    analyzer_coordination: float | None = None
    analyzer_density: float | None = None
    analyzer_wildcard: float | None = None
    pattern_stream: float | None = None
    pattern_chordstream: float | None = None
    pattern_jacks: float | None = None
    pattern_coordination: float | None = None
    pattern_density: float | None = None
    pattern_wildcard: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cond_vec_version"] = _v1_cond_version()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ChartMeta:
        expected = _v1_cond_version()
        version = data.get("cond_vec_version")
        if version != expected:
            raise ChartMetaVersionError(
                f"cond_vec_version {version!r} != {expected}"
            )
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def failed(cls, path: Path | str, error: str) -> ChartMeta:
        return cls(osu_path=str(path), error=error)


def compute_chart_meta(path: Path | str) -> ChartMeta:
    """Compute per-chart metadata. Component failures are recorded in ``error``."""
    path = Path(path)
    try:
        beatmap = parse_beatmap(path)
        canon = CanonicalTiming.from_beatmap(beatmap)
        errors: list[str] = []

        sr: float | None = None
        try:
            sr = official_star_rating(path)
        except Exception as exc:
            errors.append(f"official_sr: {exc}")

        rework: float | None = None
        try:
            from audio2map.features.cond.v1.rework import rework_star_rating

            rework = rework_star_rating(path)
        except Exception as exc:
            errors.append(f"rework_sr: {exc}")

        msd = None
        try:
            msd = compute_msd(beatmap)
        except Exception as exc:
            errors.append(f"msd: {exc}")

        pat = None
        core: dict[str, float | None] = {}
        pattern_names = (
            "Stream",
            "Chordstream",
            "Jacks",
            "Coordination",
            "Density",
            "Wildcard",
        )
        amounts: dict[str, float | None] = {name: None for name in pattern_names}
        try:
            from audio2map.features.cond.v1.patterns import (
                PATTERN_ORDER,
                analyze_patterns_once,
            )

            passed = analyze_patterns_once(path)
            pat = passed.features
            core = pat.core_dist
            if passed.pattern_amounts is not None:
                for name, value in zip(PATTERN_ORDER, passed.pattern_amounts, strict=True):
                    amounts[name] = float(value)
        except Exception as exc:
            errors.append(f"analyzer: {exc}")

        return ChartMeta(
            osu_path=str(path),
            beatmap_id=beatmap.metadata.beatmap_id,
            canonical_bpm=canon.canonical_bpm,
            official_sr=sr,
            rework_sr=rework,
            msd_overall=msd.overall if msd else None,
            msd_stream=msd.stream if msd else None,
            msd_jumpstream=msd.jumpstream if msd else None,
            msd_handstream=msd.handstream if msd else None,
            msd_stamina=msd.stamina if msd else None,
            msd_jack_speed=msd.jack_speed if msd else None,
            msd_chordjack=msd.chordjack if msd else None,
            msd_technical=msd.technical if msd else None,
            analyzer_ln_percent=pat.ln_percent if pat else None,
            analyzer_hb_row_ratio=pat.hb_row_ratio if pat else None,
            analyzer_stream=core.get("Stream"),
            analyzer_chordstream=core.get("Chordstream"),
            analyzer_jacks=core.get("Jacks"),
            analyzer_coordination=core.get("Coordination"),
            analyzer_density=core.get("Density"),
            analyzer_wildcard=core.get("Wildcard"),
            pattern_stream=amounts["Stream"],
            pattern_chordstream=amounts["Chordstream"],
            pattern_jacks=amounts["Jacks"],
            pattern_coordination=amounts["Coordination"],
            pattern_density=amounts["Density"],
            pattern_wildcard=amounts["Wildcard"],
            error="; ".join(errors) if errors else None,
        )
    except Exception as exc:
        return ChartMeta.failed(path, str(exc))


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
                meta = ChartMeta.from_dict(rec)
                out[str(Path(meta.osu_path).resolve())] = meta
            except (json.JSONDecodeError, TypeError, KeyError, ChartMetaVersionError) as exc:
                logger.warning("skip manifest line %d: %s", line_no, exc)
    logger.info("loaded %d chart meta records from %s", len(out), manifest_path)
    return out
