#!/usr/bin/env python3
"""Generate mania chart from external audio + std timing + sampled cond_vec."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import zipfile
from pathlib import Path

import torch

from audio2map.data.cond_vec import build_cond_vec
from audio2map.difficulty.chart_meta import ChartMeta
from audio2map.difficulty.chart_meta_manifest import load_chart_meta_manifest
from audio2map.osu.export import export_beatmap_notes
from audio2map.osu.parser import parse_sections, parse_timing_points
from audio2map.osu.row_tokens import CanonicalTiming
from audio2map.training.inference import OverlapConfig, generate_chart_notes
from audio2map.training.model import load_checkpoint
from audio2map.utils.paths import audio_grid_dir, processed_v2_dir


def mania_template_from_timing_osu(timing_osu: Path, out_path: Path) -> Path:
    """Clone a non-mania (or any) .osu as empty mania 4K shell for export."""
    lines = timing_osu.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    in_hit = False
    for line in lines:
        if line.startswith("Mode:"):
            out.append("Mode: 3")
            continue
        if line.startswith("CircleSize:"):
            out.append("CircleSize: 4")
            continue
        if line.startswith("[HitObjects]"):
            in_hit = True
            out.append(line)
            continue
        if in_hit:
            continue
        out.append(line)
    if not in_hit:
        out.append("")
        out.append("[HitObjects]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    return out_path


def timing_from_osu(path: Path) -> CanonicalTiming:
    sections = parse_sections(path.read_text(encoding="utf-8", errors="replace"))
    tps = parse_timing_points(sections.get("[TimingPoints]", []))
    if not tps:
        raise ValueError(f"no timing points in {path}")
    return CanonicalTiming.from_timing_points(tps)


def sample_cond_meta(
    *,
    sr_min: float,
    sr_max: float,
    seed: int,
    cond_osu: Path | None = None,
    max_ln_percent: float | None = None,
    min_stream_ratio: float | None = None,
    prefer_easy: bool = False,
    prefer_high_ln: bool = False,
) -> tuple[ChartMeta, object]:
    manifest = load_chart_meta_manifest()
    if cond_osu is not None:
        key = str(cond_osu.resolve())
        meta = manifest.get(key)
        if meta is None:
            raise SystemExit(f"cond chart not in manifest: {cond_osu}")
        return meta, build_cond_vec(meta)

    pool = [
        m
        for m in manifest.values()
        if m.error is None
        and m.official_sr is not None
        and sr_min <= m.official_sr <= sr_max
        and m.msd_overall is not None
        and (max_ln_percent is None or (m.analyzer_ln_percent or 1.0) <= max_ln_percent)
        and (min_stream_ratio is None or (m.analyzer_stream or 0.0) >= min_stream_ratio)
    ]
    if not pool:
        raise SystemExit(
            f"no manifest charts with SR in [{sr_min}, {sr_max}]"
            + (f" and ln<={max_ln_percent}" if max_ln_percent is not None else "")
            + (f" and stream>={min_stream_ratio}" if min_stream_ratio is not None else "")
        )
    if prefer_high_ln:
        pool.sort(key=lambda m: (-(m.analyzer_ln_percent or 0.0), m.official_sr or 0.0))
        meta = pool[0]
        return meta, build_cond_vec(meta)
    if prefer_easy:
        pool.sort(
            key=lambda m: (
                m.msd_overall or 99.0,
                m.analyzer_density or 1.0,
                m.analyzer_ln_percent or 1.0,
                m.official_sr or 0.0,
            )
        )
        meta = pool[0]
        return meta, build_cond_vec(meta)
    if min_stream_ratio is not None:
        pool.sort(key=lambda m: (-(m.analyzer_stream or 0.0), m.analyzer_ln_percent or 1.0))
        meta = pool[0]
        return meta, build_cond_vec(meta)
    if max_ln_percent is not None:
        pool.sort(key=lambda m: (m.analyzer_ln_percent or 1.0, m.official_sr or 0.0))
        meta = pool[0]
        return meta, build_cond_vec(meta)

    rng = random.Random(seed)
    meta = rng.choice(pool)
    cond = build_cond_vec(meta)
    return meta, cond


def pack_osz(osu_path: Path, audio_path: Path, out_osz: Path, *, extra_files: list[Path] | None = None) -> None:
    out_osz.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_osz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(osu_path, osu_path.name)
        zf.write(audio_path, audio_path.name)
        for extra in extra_files or []:
            if extra.is_file():
                zf.write(extra, extra.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osz", type=str, default=None, help="source .osz (extract if --work-dir empty)")
    p.add_argument("--work-dir", type=str, default=None, help="extracted set folder with audio + .osu")
    p.add_argument(
        "--timing-osu",
        type=str,
        default=None,
        help="std (or any) .osu for [TimingPoints] (default: first .osu in work-dir)",
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--sr-min", type=float, default=5.0)
    p.add_argument("--sr-max", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cond-osu",
        type=str,
        default=None,
        help="use this chart's cond_vec instead of random SR sample",
    )
    p.add_argument(
        "--max-ln-percent",
        type=float,
        default=None,
        help="when sampling, pick lowest analyzer_ln_percent within SR range (<= this cap)",
    )
    p.add_argument(
        "--min-stream-ratio",
        type=float,
        default=None,
        help="when sampling, pick highest analyzer_stream within SR range (>= this floor)",
    )
    p.add_argument(
        "--prefer-high-ln",
        action="store_true",
        help="pick highest analyzer_ln_percent within SR range (LN-heavy chart style)",
    )
    p.add_argument(
        "--prefer-easy",
        action="store_true",
        help="pick lowest MSD + density within SR range (simpler chart style)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("infer_custom_osz")

    if args.work_dir:
        work_dir = Path(args.work_dir)
    elif args.osz:
        work_dir = processed_v2_dir() / "infer_samples" / f"custom_{Path(args.osz).stem.replace(' ', '_')[:40]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        import zipfile as zf_mod

        with zf_mod.ZipFile(args.osz, "r") as zf:
            zf.extractall(work_dir)
        log.info("extracted %s -> %s", args.osz, work_dir)
    else:
        raise SystemExit("provide --osz or --work-dir")

    mp3s = list(work_dir.glob("*.mp3"))
    if len(mp3s) != 1:
        raise SystemExit(f"expected one mp3 in {work_dir}, found {len(mp3s)}")
    audio_path = mp3s[0]

    if args.timing_osu:
        timing_osu = Path(args.timing_osu)
    else:
        osus = sorted(work_dir.glob("*.osu"))
        if not osus:
            raise SystemExit(f"no .osu in {work_dir}")
        timing_osu = osus[0]
        for candidate in osus:
            if "Insane" in candidate.name or "Extra" in candidate.name:
                timing_osu = candidate
                break

    timing = timing_from_osu(timing_osu)
    cond_meta, cond_vec = sample_cond_meta(
        sr_min=args.sr_min,
        sr_max=args.sr_max,
        seed=args.seed,
        cond_osu=Path(args.cond_osu) if args.cond_osu else None,
        max_ln_percent=args.max_ln_percent,
        min_stream_ratio=args.min_stream_ratio,
        prefer_easy=args.prefer_easy,
        prefer_high_ln=args.prefer_high_ln,
    )

    out_dir = Path(args.out_dir) if args.out_dir else work_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.checkpoint), device)
    temperature = 0.0 if args.greedy else args.temperature

    log.info(
        "cond donor: sr=%.2f msd=%.1f ln=%.3f stream=%.3f dens=%.3f %s",
        cond_meta.official_sr,
        cond_meta.msd_overall or 0.0,
        cond_meta.analyzer_ln_percent or 0.0,
        cond_meta.analyzer_stream or 0.0,
        cond_meta.analyzer_density or 0.0,
        Path(cond_meta.osu_path).name,
    )
    log.info("timing from: %s", timing_osu.name)
    log.info("audio: %s (canonical_bpm=%.1f offset=%d)", audio_path.name, timing.canonical_bpm, timing.offset_ms)

    template = mania_template_from_timing_osu(timing_osu, out_dir / "_mania_template.osu")
    notes, report = generate_chart_notes(
        model,
        audio_path=timing_osu,
        timing=timing,
        cond_vec=cond_vec,
        grid_dir=audio_grid_dir(),
        overlap=OverlapConfig(),
        device=device,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        return_report=True,
    )

    title_stub = timing_osu.stem[:80]
    gen_osu = out_dir / f"{title_stub} [Audio2Map].osu"
    export_beatmap_notes(template, notes, gen_osu)

    extras = [p for p in work_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    osz_path = out_dir / f"{work_dir.name}_Audio2Map.osz"
    pack_osz(gen_osu, audio_path, osz_path, extra_files=extras)

    summary = {
        "work_dir": str(work_dir),
        "timing_osu": str(timing_osu),
        "audio": str(audio_path),
        "generated_osu": str(gen_osu),
        "osz": str(osz_path),
        "checkpoint": args.checkpoint,
        "cond_donor": {
            "osu_path": cond_meta.osu_path,
            "official_sr": cond_meta.official_sr,
            "version": cond_meta.version,
        },
        "cond_vec": cond_vec.tolist(),
        "generation": report.to_dict(),
        "note_count": len(notes),
        "decode": {"temperature": temperature, "top_p": args.top_p, "top_k": args.top_k},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
