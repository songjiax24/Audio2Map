"""``audio2map-infer-v1``: sample one v1 chart and write ``.osu``.

Inputs are an audio file, a timing file, and a condition JSON. Decoding and
writing stay here. This is not ``audio2map-infer`` (v0 16-bar stitch) and not
``teacher_decode`` (gold prefix). A missing checkpoint uses the initialized
model so the path can be smoked; that chart is not a trained result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audio2map.cli.common import setup_logging
from audio2map.features.audio.v1.decode import decode_audio
from audio2map.generate.v1.chart import (
    condition_from_json,
    generate_v1_chart,
    load_v1_model,
)
from audio2map.generate.v1.notes import hit_lines_from_notes
from audio2map.generate.v1.stream import KEEP_FRAMES
from audio2map.osu.v1.export import write_v1_osu
from audio2map.osu.v1.timing import parse_redlines_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a v1 chart and write .osu")
    parser.add_argument("--audio", type=str, required=True, help="audio file")
    parser.add_argument(
        "--timing",
        type=str,
        required=True,
        help="file with [TimingPoints]; only red lines are read",
    )
    parser.add_argument(
        "--cond",
        type=str,
        required=True,
        help="ChartMeta JSON (cond_vec_version 3)",
    )
    parser.add_argument("--out", type=str, required=True, help="output .osu path")
    parser.add_argument("--title", type=str, default="Untitled")
    parser.add_argument("--artist", type=str, default="Unknown Artist")
    parser.add_argument("--creator", type=str, default="Audio2Map")
    parser.add_argument("--version", type=str, default="Audio2Map")
    parser.add_argument("--checkpoint", type=str, default=None, help="v1 step_N.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--keep-frames", type=int, default=KEEP_FRAMES, dest="keep_frames")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    audio = Path(args.audio)
    timing = Path(args.timing)
    cond_path = Path(args.cond)
    out = Path(args.out)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    decoded = decode_audio(audio)
    if decoded.timeline_spliced:
        raise ValueError(f"音频损坏: {audio}")
    redlines = parse_redlines_file(timing).redlines
    model = load_v1_model(checkpoint, device)
    chart = generate_v1_chart(
        model,
        decoded.y,
        redlines,
        condition_from_json(cond_path),
        device=device,
        seed=args.seed,
        temperature=args.temperature,
        keep_frames=args.keep_frames,
    )
    audit = write_v1_osu(
        out,
        hit_lines_from_notes(chart.notes),
        list(chart.warnings),
        redlines,
        audio_filename=audio.name,
        title=args.title,
        artist=args.artist,
        creator=args.creator,
        version=args.version,
    )
    summary = {
        "audio": str(audio),
        "timing": str(timing),
        "cond": str(cond_path),
        "output": str(audit.path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "tokens": len(chart.tokens),
        "notes": len(chart.notes),
        "frames": chart.frames,
        "segments": chart.segments,
        "hit_lines": audit.hit_lines,
        "warnings": list(chart.warnings),
        "seed": args.seed,
        "temperature": args.temperature,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
