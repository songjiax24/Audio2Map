#!/usr/bin/env python3
"""Visualize one v2 training window: tick log-mel + mania chart (PNG)."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from audio2map.data.audio_grid import compute_audio_grid, slice_audio_window
from audio2map.data.chart_bundle import ChartBundle
from audio2map.data.cond_vec import build_cond_vec
from audio2map.data.v2_dataset import V2TrainingSample
from audio2map.data.window_sampler import (
    WindowSamplingConfig,
    audio_bar_range_from_duration,
    build_loss_mask,
    chart_bar_range,
    train_bar_range,
)
from audio2map.difficulty.chart_meta import compute_chart_meta
from audio2map.osu.grid_config import TICKS_PER_BAR, WINDOW_BARS
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.row_tokens import (
    TOKEN_BAR,
    TOKEN_ROW_PREFIX,
    REAL_EVENT_STATES,
    CanonicalTiming,
    beatmap_to_window_tokens,
    build_vocab,
    row_state_from_token,
)
from audio2map.utils.paths import raw_dir

LANE_COLORS = {
    1: np.array([79, 195, 247], dtype=np.uint8),
    2: np.array([129, 199, 132], dtype=np.uint8),
    4: np.array([255, 183, 77], dtype=np.uint8),
}


def _events_from_tokens(tokens: list[str], start_bar: int, end_bar: int) -> list[tuple[int, int, int]]:
    bar = start_bar - 1
    pos: int | None = None
    out: list[tuple[int, int, int]] = []
    for tok in tokens:
        if tok == TOKEN_BAR:
            bar += 1
            pos = None
        elif tok.startswith("<POS_"):
            pos = int(tok[5:-1])
        elif tok.startswith(TOKEN_ROW_PREFIX) and pos is not None and start_bar <= bar < end_bar:
            row = row_state_from_token(tok)
            tick = bar * TICKS_PER_BAR + pos
            for col, st in enumerate(row):
                if st in REAL_EVENT_STATES:
                    out.append((tick, col, int(st)))
    return out


def _db_to_rgb(v: float) -> np.ndarray:
    t = float(np.clip((v + 80.0) / 80.0, 0.0, 1.0))
    return np.array([20 + 180 * t, 10 + 90 * t * t, 40 + 120 * (1.0 - t)], dtype=np.uint8)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Write ``(H, W, 3)`` uint8 RGB array as PNG (stdlib only)."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint8 RGB")
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def render_training_window_png(
    sample,
    *,
    bundle: ChartBundle,
    summary: str,
    scale: int = 2,
) -> np.ndarray:
    feats = sample.audio_features
    start_bar = sample.window["start_bar"]
    window_bars = sample.window["window_bars"]
    end_bar = start_bar + window_bars
    start_tick = start_bar * TICKS_PER_BAR

    n_ticks, n_mels = feats.shape
    spec_h, spec_w = n_mels * scale, n_ticks * scale
    chart_h, chart_w = 96 * scale, n_ticks * scale
    margin = 8 * scale
    text_h = 72 * scale
    total_h = text_h + margin + spec_h + margin + chart_h + margin
    total_w = max(spec_w, chart_w) + 2 * margin

    img = np.full((total_h, total_w, 3), 15, dtype=np.uint8)

    # Spectrogram panel
    spec_top = text_h + margin
    spec_left = margin
    spec_rgb = np.zeros((n_mels, n_ticks, 3), dtype=np.uint8)
    for t in range(n_ticks):
        for m in range(n_mels):
            spec_rgb[n_mels - 1 - m, t] = _db_to_rgb(float(feats[t, m]))
    spec_up = np.repeat(np.repeat(spec_rgb, scale, axis=0), scale, axis=1)
    img[spec_top : spec_top + spec_h, spec_left : spec_left + spec_w] = spec_up

    for b in range(window_bars + 1):
        x = spec_left + b * TICKS_PER_BAR * scale
        img[spec_top : spec_top + spec_h, x : x + scale] = [255, 255, 255]

    # Chart panel
    chart_top = spec_top + spec_h + margin
    chart_left = margin
    panel = np.full((chart_h, chart_w, 3), 17, dtype=np.uint8)
    lane_h = chart_h // 4
    for lane in range(4):
        y = lane * lane_h + lane_h // 2
        panel[y : y + 1, :] = [48, 48, 48]

    for tick, col, st in _events_from_tokens(sample.tokens, start_bar, end_bar):
        x = (tick - start_tick) * scale
        if not (0 <= x < chart_w):
            continue
        y_center = col * lane_h + lane_h // 2
        color = LANE_COLORS.get(st, np.array([255, 255, 255], dtype=np.uint8))
        if st == 4:
            y0 = max(0, y_center - 6 * scale)
            y1 = min(chart_h, y_center + 6 * scale)
            x0 = max(0, x - 3 * scale)
            x1 = min(chart_w, x + 3 * scale)
            panel[y0:y1, x0:x1] = color
        else:
            r = (5 if st == 1 else 6) * scale
            y0, y1 = max(0, y_center - r), min(chart_h, y_center + r + 1)
            x0, x1 = max(0, x - r), min(chart_w, x + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (yy - y_center) ** 2 + (xx - x) ** 2 <= r * r
            panel[y0:y1, x0:x1][mask] = color

    for b in range(window_bars + 1):
        x = b * TICKS_PER_BAR * scale
        panel[:, x : x + 1] = [64, 64, 64]

    img[chart_top : chart_top + chart_h, chart_left : chart_left + chart_w] = panel
    return img


def _count_event_rows(tokens: list[str]) -> int:
    return sum(
        1
        for t in tokens
        if t.startswith(TOKEN_ROW_PREFIX)
        and any(s in REAL_EVENT_STATES for s in row_state_from_token(t))
    )


def build_window_sample(
    bundle: ChartBundle,
    *,
    start_bar: int,
    window_bars: int,
) -> V2TrainingSample:
    end_bar = start_bar + window_bars
    tokens = beatmap_to_window_tokens(
        bundle.beatmap,
        start_bar=start_bar,
        window_bars=window_bars,
        timing=bundle.timing,
    )
    vocab = build_vocab()
    audio_features, slice_info = slice_audio_window(
        bundle.grid, bundle.grid_meta, start_bar, end_bar
    )
    cs, ce = chart_bar_range(bundle.beatmap, bundle.timing)
    as_, ae = audio_bar_range_from_duration(bundle.grid_meta.duration_ms, bundle.timing)
    return V2TrainingSample(
        set_id=bundle.beatmap.metadata.beatmap_set_id,
        beatmap_id=bundle.beatmap.metadata.beatmap_id,
        osu_path=str(bundle.path),
        window={"start_bar": start_bar, "window_bars": window_bars},
        tokens=tokens,
        token_ids=[vocab[t] for t in tokens],
        loss_mask=build_loss_mask(tokens),
        cond_vec=bundle.cond_vec,
        audio_features=audio_features,
        audio_slice={
            "start_tick": slice_info["window_start_tick"],
            "length_ticks": window_bars * TICKS_PER_BAR,
            "feature_dim": bundle.grid_meta.feature_dim,
        },
        meta={
            "chart_start_bar": cs,
            "chart_end_bar": ce,
            "audio_start_bar": as_,
            "audio_end_bar": ae,
        },
    )


def audio_window_starts(
    audio_start: int,
    audio_end: int,
    *,
    window_bars: int,
    count: int,
) -> list[int]:
    """Evenly spaced window starts across full audio ``[audio_start, audio_end)``."""
    last_start = audio_end - window_bars
    if last_start < audio_start:
        return []
    if count <= 1:
        return [audio_start]
    span = last_start - audio_start
    return [audio_start + (span * i) // (count - 1) for i in range(count)]


def load_bundle(set_id: int) -> ChartBundle:
    osu_path = _resolve_osu(set_id)
    beatmap = parse_beatmap(osu_path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    grid, grid_meta = compute_audio_grid(osu_path.parent / "audio.mp3", timing)
    return ChartBundle(
        path=osu_path,
        beatmap=beatmap,
        timing=timing,
        cond_vec=build_cond_vec(compute_chart_meta(osu_path)),
        grid=grid,
        grid_meta=grid_meta,
    )


def export_window_png(
    bundle: ChartBundle,
    sample: V2TrainingSample,
    *,
    out: Path,
    scale: int,
    set_id: int,
) -> tuple[str, int]:
    events = _count_event_rows(sample.tokens)
    w0 = sample.window["start_bar"]
    w1 = w0 + sample.window["window_bars"]
    cs, ce = chart_bar_range(bundle.beatmap, bundle.timing)
    as_ = sample.meta["audio_start_bar"]
    ae = sample.meta["audio_end_bar"]
    in_chart = w0 < ce and w1 > cs
    summary = (
        f"set={set_id} {bundle.beatmap.metadata.version} | "
        f"window [{w0},{w1}) events={events} in_chart={in_chart} | "
        f"chart [{cs},{ce}) audio [{as_},{ae})"
    )
    rgb = render_training_window_png(sample, bundle=bundle, summary=summary, scale=scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_png(out, rgb)
    return summary, events


def _resolve_osu(set_id: int) -> Path:
    set_dir = raw_dir() / str(set_id)
    osu_paths = sorted(set_dir.glob("*.osu"))
    if not osu_paths:
        raise FileNotFoundError(f"no .osu under {set_dir}")
    if len(osu_paths) == 1:
        return osu_paths[0]
    for cand in osu_paths:
        if "autophobia" in cand.stem.lower():
            return cand
    return osu_paths[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Export training window PNG(s)")
    p.add_argument("--set-id", type=int, default=1010164)
    p.add_argument("--window-bars", type=int, default=WINDOW_BARS)
    p.add_argument("--scale", type=int, default=2, help="pixels per tick/mel band")
    p.add_argument("--count", type=int, default=1, help="number of windows (evenly spaced)")
    p.add_argument(
        "--range",
        choices=("audio", "train"),
        default="audio",
        help="audio=full song bar range; train=chart±margin sampling range",
    )
    p.add_argument("--start-bar", type=int, default=None, help="single fixed window start bar")
    p.add_argument("--out", type=Path, default=None, help="output path (single window only)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory for batch (default: viz/<set_id>/)",
    )
    args = p.parse_args()

    bundle = load_bundle(args.set_id)
    cs, ce = chart_bar_range(bundle.beatmap, bundle.timing)
    as_, ae = audio_bar_range_from_duration(bundle.grid_meta.duration_ms, bundle.timing)
    cfg = WindowSamplingConfig()
    ts, te = train_bar_range(
        audio_start_bar=as_,
        audio_end_bar=ae,
    )

    if args.start_bar is not None:
        starts = [args.start_bar]
    elif args.range == "audio":
        starts = audio_window_starts(as_, ae, window_bars=args.window_bars, count=args.count)
    else:
        last = te - args.window_bars
        starts = audio_window_starts(ts, te, window_bars=args.window_bars, count=args.count) if last >= ts else []

    if not starts:
        raise SystemExit("no valid window starts for given range/count")

    out_dir = args.out_dir or (Path(raw_dir()).parent / "viz" / str(args.set_id))
    for i, start_bar in enumerate(starts):
        sample = build_window_sample(bundle, start_bar=start_bar, window_bars=args.window_bars)
        events = _count_event_rows(sample.tokens)
        if len(starts) == 1 and args.out is not None:
            out = args.out
        else:
            tag = f"{i:02d}_bar{start_bar}_e{events}" if len(starts) > 1 else f"{args.set_id}_bar{start_bar}_e{events}"
            out = out_dir / f"{tag}.png"
        summary, _ = export_window_png(
            bundle,
            sample,
            out=out,
            scale=args.scale,
            set_id=args.set_id,
        )
        print(out)
        print(summary)


if __name__ == "__main__":
    main()
