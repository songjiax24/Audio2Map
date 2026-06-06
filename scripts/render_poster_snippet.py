#!/usr/bin/env python3
"""Render a poster snippet: log-mel (top) + mania chart (bottom), transparent PNG."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from audio2map.audio.loader import load_mono_audio
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.schema import Beatmap, ManiaNote, NoteType, TimingPoint


def _parse_time(s: str) -> float:
    """Parse ``M:SS``, ``M:SS.s``, or plain seconds."""
    s = s.strip()
    if ":" in s:
        m, rest = s.split(":", 1)
        return float(m) * 60.0 + float(rest)
    return float(s)


def _write_png_rgba(path: Path, rgba: np.ndarray) -> None:
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("expected (H, W, 4) uint8 RGBA")
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _active_timing_at(timing_points: list[TimingPoint], t_ms: float) -> tuple[int, float, int]:
    """Return ``(origin_ms, beat_length_ms, meter)`` active at ``t_ms``."""
    origin = 0
    beat_length = 500.0
    meter = 4
    for tp in timing_points:
        if tp.offset_ms > t_ms:
            break
        if tp.uninherited:
            origin = tp.offset_ms
            beat_length = tp.beat_length_ms
            meter = tp.meter
    return origin, beat_length, meter


def _grid_times_ms(
    timing_points: list[TimingPoint],
    *,
    start_ms: float,
    end_ms: float,
) -> tuple[list[float], list[float]]:
    """Return beat and bar (measure) line times in ``[start_ms, end_ms]``."""
    origin, beat_len, meter = _active_timing_at(timing_points, start_ms)
    bar_len = beat_len * meter
    beats: list[float] = []
    bars: list[float] = []
    if beat_len <= 0 or bar_len <= 0:
        return beats, bars
    # floor so a window clipped exactly to a bar/beat boundary still gets the left grid line
    n0 = int(np.floor((start_ms - origin) / beat_len))
    t = origin + n0 * beat_len
    while t <= end_ms + 1e-6:
        if start_ms - 1e-6 <= t <= end_ms + 1e-6:
            beats.append(t)
        t += beat_len
    m0 = int(np.floor((start_ms - origin) / bar_len))
    t = origin + m0 * bar_len
    while t <= end_ms + 1e-6:
        if start_ms - 1e-6 <= t <= end_ms + 1e-6:
            bars.append(t)
        t += bar_len
    return beats, bars


def _draw_vline(img: np.ndarray, x: float, color: tuple[int, int, int, int], *, width: int = 1) -> None:
    xi = int(round(x))
    for dx in range(width):
        xpos = xi + dx
        if 0 <= xpos < img.shape[1]:
            for y in range(img.shape[0]):
                _blend_pixel(img, y, xpos, color)


def _overlay_grid_lines(
    img: np.ndarray,
    *,
    beats: list[float],
    bars: list[float],
    start_ms: float,
    end_ms: float,
) -> None:
    w = img.shape[1]
    for t in beats:
        x = _x_for_ms(t, start_ms=start_ms, end_ms=end_ms, width=w)
        _draw_vline(img, x, (255, 255, 255, 28))
    for t in bars:
        x = _x_for_ms(t, start_ms=start_ms, end_ms=end_ms, width=w)
        _draw_vline(img, x, (255, 255, 255, 110), width=2)


def _magma_rgba(db: float, *, vmin: float, vmax: float, alpha_gamma: float = 0.65) -> tuple[int, int, int, int]:
    t = float(np.clip((db - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0))
    if t < 0.04:
        return (0, 0, 0, 0)
    r = int(20 + 180 * t)
    g = int(10 + 90 * t * t)
    b = int(40 + 120 * (1.0 - t))
    a = int(255 * (t**alpha_gamma))
    return (r, g, b, max(a, 0))


def _render_log_mel_rgba(
    y: np.ndarray,
    sr: int,
    *,
    start_ms: float,
    end_ms: float,
    width: int,
    height: int,
) -> np.ndarray:
    import librosa

    hop = 128
    n_fft = 1024
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=128, fmin=20.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    hop_ms = hop * 1000.0 / sr
    i0 = max(0, int(np.floor(start_ms / hop_ms)))
    i1 = min(len(log_mel), int(np.ceil(end_ms / hop_ms)))
    chunk = log_mel[i0:i1]
    if chunk.shape[0] < 2:
        chunk = np.full((2, 128), -80.0, dtype=np.float32)

    resampled = np.empty((width, 128), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, chunk.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, width, endpoint=False)
    for band in range(128):
        resampled[:, band] = np.interp(x_new, x_old, chunk[:, band])

    vmin = float(np.percentile(resampled, 5))
    vmax = float(np.percentile(resampled, 99))
    if vmax - vmin < 8.0:
        vmax = vmin + 8.0

    out = np.zeros((height, width, 4), dtype=np.uint8)
    band_h = height / 128.0
    for x in range(width):
        for band in range(128):
            y0 = int((127 - band) * band_h)
            y1 = max(y0 + 1, int((128 - band) * band_h))
            out[y0:y1, x] = _magma_rgba(float(resampled[x, band]), vmin=vmin, vmax=vmax)
    return out


def _window_last_bars(
    timing_points: list[TimingPoint],
    *,
    anchor_start_ms: float,
    anchor_end_ms: float,
    n_bars: int,
) -> tuple[float, float]:
    """Clip to the last ``n_bars`` full measures inside ``[anchor_start, anchor_end]``."""
    _, bars = _grid_times_ms(
        timing_points, start_ms=anchor_start_ms, end_ms=anchor_end_ms,
    )
    if len(bars) < n_bars + 1:
        if len(bars) >= 2:
            return bars[-(n_bars + 1)], min(anchor_end_ms, bars[-1])
        return anchor_start_ms, anchor_end_ms
    return bars[-(n_bars + 1)], min(anchor_end_ms, bars[-1])


def _x_for_ms(t_ms: float, *, start_ms: float, end_ms: float, width: int) -> float:
    return (t_ms - start_ms) / (end_ms - start_ms) * width


def _blend_pixel(img: np.ndarray, y: int, x: int, color: tuple[int, int, int, int]) -> None:
    if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
        return
    cr, cg, cb, ca = color
    if ca <= 0:
        return
    pr, pg, pb, pa = img[y, x]
    if pa == 0:
        img[y, x] = (cr, cg, cb, ca)
        return
    a = ca / 255.0
    inv = 1.0 - a
    img[y, x] = (
        int(cr * a + pr * inv),
        int(cg * a + pg * inv),
        int(cb * a + pb * inv),
        int(ca + pa * inv),
    )


def _draw_disk(img: np.ndarray, cx: float, cy: float, r: float, color: tuple[int, int, int, int]) -> None:
    y0, y1 = int(cy - r - 1), int(cy + r + 2)
    x0, x1 = int(cx - r - 1), int(cx + r + 2)
    for y in range(max(0, y0), min(img.shape[0], y1)):
        for x in range(max(0, x0), min(img.shape[1], x1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                _blend_pixel(img, y, x, color)


def _draw_rect_alpha(
    img: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: tuple[int, int, int, int],
) -> None:
    xi0, xi1 = int(x0), int(np.ceil(x1))
    yi0, yi1 = int(y0), int(np.ceil(y1))
    for y in range(max(0, yi0), min(img.shape[0], yi1)):
        for x in range(max(0, xi0), min(img.shape[1], xi1)):
            _blend_pixel(img, y, x, color)


def _render_chart_rgba(
    notes: list[ManiaNote],
    *,
    start_ms: float,
    end_ms: float,
    width: int,
    height: int,
) -> np.ndarray:
    img = np.zeros((height, width, 4), dtype=np.uint8)
    lane_h = height / 4.0
    tap_r = lane_h * 0.18
    tap_rgb = (160, 190, 200, 235)  # #A0BEC8
    hold_rgb = (120, 220, 140, 200)
    hold_end_rgb = (255, 180, 80, 220)
    hold_body_rgb = (120, 220, 140, 90)

    # lane guides (very subtle)
    for lane in range(1, 4):
        y = int(lane * lane_h)
        for x in range(width):
            _blend_pixel(img, y, x, (255, 255, 255, 18))

    for note in sorted(notes, key=lambda n: (n.time_ms, n.col)):
        if note.note_type == NoteType.HOLD and note.end_time_ms is not None:
            if note.end_time_ms < start_ms or note.time_ms > end_ms:
                continue
            x0 = _x_for_ms(max(note.time_ms, start_ms), start_ms=start_ms, end_ms=end_ms, width=width)
            x1 = _x_for_ms(min(note.end_time_ms, end_ms), start_ms=start_ms, end_ms=end_ms, width=width)
            y0 = note.col * lane_h + lane_h * 0.22
            y1 = note.col * lane_h + lane_h * 0.78
            _draw_rect_alpha(img, x0, x1, y0, y1, hold_body_rgb)
            cx = _x_for_ms(note.time_ms, start_ms=start_ms, end_ms=end_ms, width=width)
            cy = note.col * lane_h + lane_h * 0.5
            _draw_disk(img, cx, cy, tap_r, hold_rgb)
            if note.end_time_ms <= end_ms:
                cx2 = _x_for_ms(note.end_time_ms, start_ms=start_ms, end_ms=end_ms, width=width)
                _draw_disk(img, cx2, cy, tap_r, hold_end_rgb)
        elif note.note_type != NoteType.HOLD:
            if not (start_ms <= note.time_ms < end_ms):
                continue
            cx = _x_for_ms(note.time_ms, start_ms=start_ms, end_ms=end_ms, width=width)
            cy = note.col * lane_h + lane_h * 0.5
            _draw_disk(img, cx, cy, tap_r, tap_rgb)

    return img


def render_poster_snippet(
    beatmap: Beatmap,
    y: np.ndarray,
    sr: int,
    *,
    start_ms: float,
    end_ms: float,
    width: int = 1400,
    spec_height: int = 420,
    chart_height: int = 220,
    gap: int = 28,
) -> np.ndarray:
    notes = [
        n
        for n in beatmap.notes
        if n.time_ms < end_ms
        and (n.end_time_ms if n.note_type == NoteType.HOLD and n.end_time_ms else n.time_ms) >= start_ms
    ]

    beats, bars = _grid_times_ms(beatmap.timing_points, start_ms=start_ms, end_ms=end_ms)

    spec = _render_log_mel_rgba(
        y, sr, start_ms=start_ms, end_ms=end_ms, width=width, height=spec_height,
    )
    chart = _render_chart_rgba(
        notes, start_ms=start_ms, end_ms=end_ms, width=width, height=chart_height,
    )

    _overlay_grid_lines(spec, beats=beats, bars=bars, start_ms=start_ms, end_ms=end_ms)
    _overlay_grid_lines(chart, beats=beats, bars=bars, start_ms=start_ms, end_ms=end_ms)

    total_h = spec_height + gap + chart_height
    out = np.zeros((total_h, width, 4), dtype=np.uint8)
    out[:spec_height] = spec
    out[spec_height + gap :] = chart
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Poster snippet: log-mel + chart, transparent PNG")
    p.add_argument("--osu", type=Path, required=True)
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--start", type=str, default="1:22", help="start time M:SS")
    p.add_argument("--end", type=str, default="1:27.6", help="end time M:SS")
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--last-bars", type=int, default=None, help="keep last N measures inside start/end")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    start_ms = _parse_time(args.start) * 1000.0
    end_ms = _parse_time(args.end) * 1000.0
    audio = args.audio or args.osu.parent / "audio.mp3"
    y, sr = load_mono_audio(audio, sample_rate=22050)

    beatmap = parse_beatmap(args.osu)
    if args.last_bars is not None:
        start_ms, end_ms = _window_last_bars(
            beatmap.timing_points,
            anchor_start_ms=start_ms,
            anchor_end_ms=end_ms,
            n_bars=args.last_bars,
        )
    rgba = render_poster_snippet(
        beatmap,
        y,
        sr,
        start_ms=start_ms,
        end_ms=end_ms,
        width=args.width,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_png_rgba(args.out, rgba)
    print(args.out, f"size={rgba.shape[1]}x{rgba.shape[0]}  range={start_ms/1000:.1f}s–{end_ms/1000:.1f}s")


if __name__ == "__main__":
    main()
