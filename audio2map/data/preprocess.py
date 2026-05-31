"""Preprocess raw beatmap sets into training-ready artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from audio2map.audio.features import align_num_frames, compute_mel
from audio2map.audio.loader import find_audio_file, load_mono_audio
from audio2map.data.events_io import events_to_arrays
from audio2map.data.selection import select_highest_od
from audio2map.osu.events import beatmap_to_events, count_event_types
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.timing import summarize_beatmap_timing

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    hop_ms: int = 10
    sample_rate: int = 44_100
    n_mels: int = 80
    mel_dtype: str = "float16"  # float16 | float32


@dataclass
class SampleMeta:
    sid: str
    osu_path: str
    audio_path: str
    title: str
    artist: str
    version: str
    od: float
    hp: float
    ar: float
    cs: float
    bpm: float | None
    constant_bpm: bool
    hop_ms: int
    sample_rate: int
    n_mels: int
    num_frames: int
    n_events: int
    n_taps: int
    n_holds: int
    ln_ratio: float
    notes_per_second: float
    npz_path: str


def _chart_max_frame(events: list) -> int:
    if not events:
        return 0
    return max(e.end_frame if e.end_frame is not None else e.frame for e in events)


def process_beatmap_set(
    set_dir: Path,
    out_dir: Path,
    cfg: PreprocessConfig,
) -> SampleMeta:
    """Preprocess one ``raw/{sid}/`` folder. Raises on failure."""
    sid = set_dir.name
    if not sid.isdigit():
        raise ValueError(f"invalid set id directory: {set_dir}")

    osu_files = sorted(set_dir.glob("*.osu"))
    osu_path = select_highest_od(osu_files)
    audio_path = find_audio_file(set_dir)

    bm = parse_beatmap(osu_path)
    events = beatmap_to_events(bm, hop_ms=cfg.hop_ms)
    counts = count_event_types(events)
    timing = summarize_beatmap_timing(bm)

    y, sr = load_mono_audio(audio_path, sample_rate=cfg.sample_rate)
    mel = compute_mel(y, sr, hop_ms=cfg.hop_ms, n_mels=cfg.n_mels)
    mel_frames = mel.shape[0]
    chart_frames = _chart_max_frame(events) + 1
    num_frames = align_num_frames(mel_frames, chart_frames)

    if mel_frames < num_frames:
        pad = np.zeros((num_frames - mel_frames, cfg.n_mels), dtype=np.float32)
        mel = np.concatenate([mel, pad], axis=0)
    elif mel_frames > num_frames:
        mel = mel[:num_frames]

    dtype = np.float16 if cfg.mel_dtype == "float16" else np.float32
    mel_store = mel.astype(dtype)

    event_arrays = events_to_arrays(events)
    duration_sec = max(num_frames * cfg.hop_ms / 1000, 1e-6)
    ln_ratio = counts["hold"] / max(counts["total"], 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sid}.npz"
    np.savez_compressed(
        npz_path,
        mel=mel_store,
        num_frames=np.int32(num_frames),
        hop_ms=np.int32(cfg.hop_ms),
        **event_arrays,
    )

    meta = SampleMeta(
        sid=sid,
        osu_path=str(osu_path),
        audio_path=str(audio_path),
        title=bm.metadata.title,
        artist=bm.metadata.artist,
        version=bm.metadata.version,
        od=bm.metadata.overall_difficulty,
        hp=bm.metadata.hp,
        ar=bm.metadata.approach_rate,
        cs=bm.metadata.circle_size,
        bpm=timing.bpm_primary,
        constant_bpm=timing.constant_bpm,
        hop_ms=cfg.hop_ms,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        num_frames=num_frames,
        n_events=counts["total"],
        n_taps=counts["tap"],
        n_holds=counts["hold"],
        ln_ratio=round(ln_ratio, 4),
        notes_per_second=round(counts["total"] / duration_sec, 4),
        npz_path=str(npz_path),
    )

    meta_path = out_dir / f"{sid}.meta.json"
    meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_sample_meta(path: Path) -> SampleMeta:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SampleMeta(**data)


def write_manifest(metas: list[SampleMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")


def iter_set_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        (p for p in raw_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def run_preprocess(
    raw_root: Path,
    out_dir: Path,
    cfg: PreprocessConfig,
    *,
    limit: int | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    set_dirs = iter_set_dirs(raw_root)
    if limit is not None:
        set_dirs = set_dirs[:limit]

    ok: list[SampleMeta] = []
    skipped = failed = 0
    errors: list[str] = []

    for set_dir in tqdm(set_dirs, desc="preprocess", unit="set"):
        sid = set_dir.name
        npz_path = out_dir / f"{sid}.npz"
        if skip_existing and npz_path.exists():
            meta_path = out_dir / f"{sid}.meta.json"
            if meta_path.exists():
                try:
                    ok.append(load_sample_meta(meta_path))
                    skipped += 1
                    continue
                except Exception:
                    pass

        try:
            meta = process_beatmap_set(set_dir, out_dir, cfg)
            ok.append(meta)
            logger.info("ok sid=%s frames=%d events=%d od=%.1f", sid, meta.num_frames, meta.n_events, meta.od)
        except Exception as exc:
            failed += 1
            msg = f"sid={sid}: {exc}"
            errors.append(msg)
            logger.exception(msg)

    manifest_path = out_dir / "manifest.jsonl"
    write_manifest(ok, manifest_path)

    return {
        "total_sets": len(set_dirs),
        "processed": len(ok) - skipped,
        "skipped_existing": skipped,
        "failed": failed,
        "manifest": str(manifest_path),
        "errors": errors[:20],
    }
