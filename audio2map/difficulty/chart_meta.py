"""Combined per-chart metadata for training conditions (v2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from audio2map.analysis.pattern_features import analyze_chart_patterns
from audio2map.data.chart_stats import hold_coverage, hold_ratio, timing_metadata
from audio2map.difficulty.minacalc import MinaCalcError, compute_msd
from audio2map.difficulty.official_sr import official_star_rating
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.osu.timing import summarize_beatmap_timing


@dataclass(slots=True)
class ChartMeta:
    osu_path: str
    set_id: int | None
    beatmap_id: int | None
    version: str
    constant_bpm: bool
    bpm: float | None
    original_bpm: float | None
    canonical_bpm: float | None
    bpm_scale_exp: int | None
    offset_ms: int | None
    meter: int | None
    note_count: int
    hold_ratio: float
    hold_coverage: float
    official_sr: float | None
    msd_overall: float | None
    msd_stream: float | None
    msd_jumpstream: float | None
    msd_handstream: float | None
    msd_stamina: float | None
    msd_jack_speed: float | None
    msd_chordjack: float | None
    msd_technical: float | None
    analyzer_ln_percent: float | None = None
    analyzer_hb_row_ratio: float | None = None
    analyzer_stream: float | None = None
    analyzer_chordstream: float | None = None
    analyzer_jacks: float | None = None
    analyzer_coordination: float | None = None
    analyzer_density: float | None = None
    analyzer_wildcard: float | None = None
    analyzer_available: int = 0
    msd_available: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_chart_meta(path: Path | str, *, skip_msd: bool = False) -> ChartMeta:
    path = Path(path)
    try:
        beatmap = parse_beatmap(path)
        timing = summarize_beatmap_timing(beatmap)
        tmeta = timing_metadata(beatmap)
        ct = CanonicalTiming.from_beatmap(beatmap)
        ln_ratio = hold_ratio(beatmap)
        ln_cov = hold_coverage(beatmap, ct)

        sr: float | None = None
        msd = None
        try:
            sr = official_star_rating(path)
        except Exception as exc:
            return _error_meta(path, beatmap, timing, tmeta, ln_ratio, ln_cov, f"official_sr: {exc}")

        if not skip_msd:
            try:
                msd = compute_msd(beatmap)
            except MinaCalcError as exc:
                return _error_meta(
                    path, beatmap, timing, tmeta, ln_ratio, ln_cov, f"msd: {exc}", sr=sr
                )

        pat = analyze_chart_patterns(path)
        core = pat.core_dist if pat.analyzer_available else {}

        return ChartMeta(
            osu_path=str(path),
            set_id=beatmap.metadata.beatmap_set_id,
            beatmap_id=beatmap.metadata.beatmap_id,
            version=beatmap.metadata.version,
            constant_bpm=timing.constant_bpm,
            bpm=timing.bpm_primary,
            original_bpm=tmeta["original_bpm"],
            canonical_bpm=tmeta["canonical_bpm"],
            bpm_scale_exp=tmeta["bpm_scale_exp"],
            offset_ms=tmeta["offset_ms"],
            meter=tmeta["meter"],
            note_count=beatmap.note_count,
            hold_ratio=ln_ratio,
            hold_coverage=ln_cov,
            official_sr=sr,
            msd_overall=msd.overall if msd else None,
            msd_stream=msd.stream if msd else None,
            msd_jumpstream=msd.jumpstream if msd else None,
            msd_handstream=msd.handstream if msd else None,
            msd_stamina=msd.stamina if msd else None,
            msd_jack_speed=msd.jack_speed if msd else None,
            msd_chordjack=msd.chordjack if msd else None,
            msd_technical=msd.technical if msd else None,
            analyzer_ln_percent=pat.ln_percent if pat.analyzer_available else None,
            analyzer_hb_row_ratio=pat.hb_row_ratio if pat.analyzer_available else None,
            analyzer_stream=core.get("Stream"),
            analyzer_chordstream=core.get("Chordstream"),
            analyzer_jacks=core.get("Jacks"),
            analyzer_coordination=core.get("Coordination"),
            analyzer_density=core.get("Density"),
            analyzer_wildcard=core.get("Wildcard"),
            analyzer_available=1 if pat.analyzer_available else 0,
            msd_available=1 if msd else 0,
        )
    except Exception as exc:
        return ChartMeta(
            osu_path=str(path),
            set_id=None,
            beatmap_id=None,
            version="",
            constant_bpm=False,
            bpm=None,
            original_bpm=None,
            canonical_bpm=None,
            bpm_scale_exp=None,
            offset_ms=None,
            meter=None,
            note_count=0,
            hold_ratio=0.0,
            hold_coverage=0.0,
            official_sr=None,
            msd_overall=None,
            msd_stream=None,
            msd_jumpstream=None,
            msd_handstream=None,
            msd_stamina=None,
            msd_jack_speed=None,
            msd_chordjack=None,
            msd_technical=None,
            error=str(exc),
        )


def _error_meta(path, beatmap, timing, tmeta, ln_ratio, ln_cov, message, *, sr=None):
    return ChartMeta(
        osu_path=str(path),
        set_id=beatmap.metadata.beatmap_set_id,
        beatmap_id=beatmap.metadata.beatmap_id,
        version=beatmap.metadata.version,
        constant_bpm=timing.constant_bpm,
        bpm=timing.bpm_primary,
        original_bpm=tmeta.get("original_bpm"),
        canonical_bpm=tmeta.get("canonical_bpm"),
        bpm_scale_exp=tmeta.get("bpm_scale_exp"),
        offset_ms=tmeta.get("offset_ms"),
        meter=tmeta.get("meter"),
        note_count=beatmap.note_count,
        hold_ratio=ln_ratio,
        hold_coverage=ln_cov,
        official_sr=sr,
        msd_overall=None,
        msd_stream=None,
        msd_jumpstream=None,
        msd_handstream=None,
        msd_stamina=None,
        msd_jack_speed=None,
        msd_chordjack=None,
        msd_technical=None,
        error=message,
    )
