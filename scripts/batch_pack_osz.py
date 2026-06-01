#!/usr/bin/env python3
"""Batch infer + pack .osz with original chart, generated chart, and audio."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from audio2map.audio.loader import find_audio_file
from audio2map.utils.paths import processed_v2_dir


def pack_compare_osz(original_osu: Path, generated_osu: Path, out_osz: Path) -> None:
    audio = find_audio_file(original_osu.parent)
    out_osz.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_osz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(original_osu, original_osu.name)
        zf.write(generated_osu, generated_osu.name)
        zf.write(audio, audio.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--osu", type=str, action="append", required=True, help="template .osu (repeatable)")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="version tag for output filenames, e.g. smoke5000_tick",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("batch_pack_osz")

    out_dir = Path(args.out_dir) if args.out_dir else processed_v2_dir() / "generated" / "compare_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_script = Path(__file__).resolve().parent / "infer_v2.py"
    results: list[dict] = []

    for osu in args.osu:
        osu_path = Path(osu)
        sid = osu_path.parent.name
        stem = osu_path.stem.replace(" ", "_")[:60]
        work = out_dir / f"{sid}_{stem}"
        work.mkdir(parents=True, exist_ok=True)

        gen_osu = work / f"{osu_path.stem} [Audio2Map].osu"
        tag_suffix = f"_{args.tag}" if args.tag else ""
        osz_name = f"compare_{sid}{tag_suffix}.osz"
        osz_path = out_dir / osz_name

        log.info("infer %s", osu_path.name)
        cmd = [
            sys.executable,
            str(infer_script),
            "--checkpoint",
            args.checkpoint,
            "--osu",
            str(osu_path),
            "--out",
            str(gen_osu),
            "--device",
            args.device,
        ]
        subprocess.run(cmd, check=True)

        pack_compare_osz(osu_path, gen_osu, osz_path)
        easy_name = f"Audio2Map_compare_{sid}{tag_suffix}.osz"
        easy = out_dir.parent.parent.parent / easy_name
        shutil.copy2(osz_path, easy)

        from audio2map.osu.parser import parse_beatmap

        orig_n = len(parse_beatmap(osu_path).notes)
        gen_n = len(parse_beatmap(gen_osu).notes)
        row = {
            "set_id": sid,
            "original": str(osu_path),
            "generated": str(gen_osu),
            "osz": str(osz_path),
            "easy_copy": str(easy),
            "original_notes": orig_n,
            "generated_notes": gen_n,
        }
        results.append(row)
        log.info("packed %s (%d -> %d notes)", osz_path.name, orig_n, gen_n)

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
