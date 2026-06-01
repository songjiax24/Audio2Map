"""Overlap window generation and full-chart inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from audio2map.data.audio_grid import load_audio_grid, resolve_grid_stem, slice_audio_window
from audio2map.data.window_sampler import audio_bar_range_from_duration
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import (
    TOKEN_BOS,
    TOKEN_EOS,
    CanonicalTiming,
    chart_event_bar_range,
    initial_row_from_active,
    invert_vocab,
    ms_to_tick,
    split_tick,
)
from audio2map.osu.round_trip import tokens_to_notes
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.training.decode import ChartDecodeState, bos_initial_prefix
from audio2map.training.model import AudioChartModel
from audio2map.utils.paths import audio_grid_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OverlapConfig:
    window_bars: int = 8
    context_bars: int = 4
    keep_bars: int = 4
    future_bars: int = 0
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.context_bars + self.keep_bars + self.future_bars != self.window_bars:
            raise ValueError(
                f"context+keep+future must equal window_bars "
                f"({self.context_bars}+{self.keep_bars}+{self.future_bars}!={self.window_bars})"
            )


@dataclass(frozen=True, slots=True)
class GenerationRangeConfig:
    """Inference bar range. Default covers full audio (no reference-chart truncation)."""

    mode: Literal["audio_full", "reference_chart"] = "audio_full"
    pre_margin_bars: int = 0
    post_margin_bars: int = 4


@dataclass(slots=True)
class WindowGenerationLog:
    win_start: int
    win_end: int
    keep_start: int
    keep_end: int
    context_bars: int
    is_first_window: bool
    notes_kept: int


@dataclass(slots=True)
class GenerationReport:
    audio_start_bar: int
    audio_end_bar: int
    gen_start_bar: int
    gen_end_bar: int
    chart_start_bar: int | None
    chart_end_bar: int | None
    range_mode: str
    first_note_bar: int | None = None
    last_note_bar: int | None = None
    tail_note_count: int = 0
    audio_pooling: str = "tick"
    audio_ticks_per_window: int = 0
    encoder_prefix_len: int = 0
    windows: list[WindowGenerationLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "audio_start_bar": self.audio_start_bar,
            "audio_end_bar": self.audio_end_bar,
            "gen_start_bar": self.gen_start_bar,
            "gen_end_bar": self.gen_end_bar,
            "chart_start_bar": self.chart_start_bar,
            "chart_end_bar": self.chart_end_bar,
            "range_mode": self.range_mode,
            "first_note_bar": self.first_note_bar,
            "last_note_bar": self.last_note_bar,
            "tail_note_count": self.tail_note_count,
            "audio_pooling": self.audio_pooling,
            "audio_ticks_per_window": self.audio_ticks_per_window,
            "encoder_prefix_len": self.encoder_prefix_len,
            "windows": [
                {
                    "win_start": w.win_start,
                    "win_end": w.win_end,
                    "keep_start": w.keep_start,
                    "keep_end": w.keep_end,
                    "context_bars": w.context_bars,
                    "is_first_window": w.is_first_window,
                    "notes_kept": w.notes_kept,
                }
                for w in self.windows
            ],
        }


def resolve_generation_bar_range(
    *,
    audio_start_bar: int,
    audio_end_bar: int,
    chart_start_bar: int | None,
    chart_end_bar: int | None,
    cfg: GenerationRangeConfig,
) -> tuple[int, int]:
    if cfg.mode == "reference_chart" and chart_start_bar is not None and chart_end_bar is not None:
        gen_start = max(audio_start_bar, chart_start_bar - cfg.pre_margin_bars)
        gen_end = min(audio_end_bar, chart_end_bar + cfg.post_margin_bars)
        return gen_start, max(gen_start, gen_end)
    return audio_start_bar, audio_end_bar


def overlap_inference_bar_windows(
    gen_start_bar: int,
    gen_end_bar: int,
    cfg: OverlapConfig,
) -> list[tuple[int, int, int, int, int]]:
    """Return ``(win_start, win_end, keep_start, keep_end, context_bars_used)``."""
    windows: list[tuple[int, int, int, int, int]] = []
    bar = gen_start_bar
    first = True
    while bar < gen_end_bar:
        win_end = min(bar + cfg.window_bars, gen_end_bar)
        win_start = win_end - cfg.window_bars
        if win_start < gen_start_bar:
            win_start = gen_start_bar
        actual_context = 0 if first else cfg.context_bars
        keep_start = win_start + actual_context
        keep_end = min(keep_start + cfg.keep_bars, win_end)
        if keep_end > keep_start:
            windows.append((win_start, win_end, keep_start, keep_end, actual_context))
        if win_end >= gen_end_bar:
            break
        bar += cfg.keep_bars
        first = False
    return windows


def sample_next_token_id(
    logits: torch.Tensor,
    allowed: set[int],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    mask = torch.full_like(logits, float("-inf"))
    for tid in allowed:
        mask[tid] = 0.0
    logits = logits + mask
    if temperature <= 0:
        return int(logits.argmax().item())

    probs = F.softmax(logits / temperature, dim=-1)
    allowed_list = sorted(allowed)
    sub = probs[allowed_list]
    if top_k > 0 and sub.numel() > top_k:
        keep = torch.topk(sub, top_k).indices
        sub = sub[keep]
        allowed_list = [allowed_list[i] for i in keep.tolist()]
    if 0 < top_p < 1.0:
        sorted_probs, order = torch.sort(sub, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        cut = int((cum <= top_p).sum().item())
        cut = max(cut, 0)
        sub = sorted_probs[: cut + 1]
        allowed_list = [allowed_list[i] for i in order[: cut + 1].tolist()]
    sub = sub / sub.sum().clamp(min=1e-12)
    idx = int(torch.multinomial(sub, 1).item())
    return allowed_list[idx]


def initial_row_at_tick(notes: list[ManiaNote], timing: CanonicalTiming, window_start_tick: int):
    active = [False, False, False, False]
    for note in notes:
        if note.note_type != NoteType.HOLD or note.end_time_ms is None:
            continue
        head = ms_to_tick(note.time_ms, timing)
        tail = ms_to_tick(note.end_time_ms, timing)
        if head < window_start_tick < tail:
            active[note.col] = True
    return initial_row_from_active(active)


@torch.no_grad()
def generate_window_tokens(
    model: AudioChartModel,
    *,
    audio: torch.Tensor,
    cond_vec: torch.Tensor,
    initial_row,
    window_bars: int,
    device: torch.device,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    prompt_token_ids: list[int] | None = None,
) -> list[int]:
    """Generate one window token sequence (including BOS/EOS)."""
    from audio2map.osu.row_tokens import build_vocab, row_state_to_token

    vocab = build_vocab()
    id_to_token = invert_vocab(vocab)
    initial_id = vocab[row_state_to_token(initial_row)]

    if prompt_token_ids:
        token_ids = list(prompt_token_ids)
    else:
        token_ids = bos_initial_prefix(initial_id, vocab=vocab)

    decode = ChartDecodeState.from_initial_row(initial_row, window_bars=window_bars, vocab=vocab)
    for tid in token_ids[2:]:
        decode.observe(tid)

    audio_b = audio.unsqueeze(0).to(device)
    cond_b = cond_vec.unsqueeze(0).to(device)

    while not decode.finished and len(token_ids) < model.max_seq_len:
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        logits = model.next_token_logits(audio_b, cond_b, ids)[0]
        allowed = decode.allowed_token_ids()
        if not allowed:
            break
        next_id = sample_next_token_id(
            logits,
            allowed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_ids.append(next_id)
        decode.observe_and_advance_bar_if_needed(next_id)
        if id_to_token[next_id] == TOKEN_EOS:
            break

    if token_ids[-1] != vocab[TOKEN_EOS]:
        token_ids.append(vocab[TOKEN_EOS])
    return token_ids


def filter_notes_by_tick_range(
    notes: list[ManiaNote],
    timing: CanonicalTiming,
    *,
    keep_start_tick: int,
    keep_end_tick: int,
) -> list[ManiaNote]:
    kept: list[ManiaNote] = []
    for n in notes:
        tick = ms_to_tick(n.time_ms, timing)
        if keep_start_tick <= tick < keep_end_tick:
            kept.append(n)
    return kept


def dedupe_notes(notes: list[ManiaNote]) -> list[ManiaNote]:
    seen: set[tuple] = set()
    out: list[ManiaNote] = []
    for n in sorted(notes, key=lambda x: (x.time_ms, x.col)):
        key = (n.time_ms, n.col, n.note_type, n.end_time_ms)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def _note_bar_range(notes: list[ManiaNote], timing: CanonicalTiming) -> tuple[int | None, int | None]:
    if not notes:
        return None, None
    bars = [split_tick(ms_to_tick(n.time_ms, timing))[0] for n in notes]
    return min(bars), max(bars)


@torch.no_grad()
def generate_chart_notes(
    model: AudioChartModel,
    *,
    audio_path: Path,
    timing: CanonicalTiming,
    cond_vec: np.ndarray,
    grid_dir: Path | None = None,
    overlap: OverlapConfig | None = None,
    range_cfg: GenerationRangeConfig | None = None,
    device: torch.device | None = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    single_window: bool = False,
    single_window_start_bar: int | None = None,
    reference_notes: list[ManiaNote] | None = None,
    return_report: bool = False,
) -> list[ManiaNote] | tuple[list[ManiaNote], GenerationReport]:
    """Generate chart notes. Default range = full audio (not reference-chart clipped)."""
    from audio2map.audio.loader import find_audio_file
    from audio2map.osu.parser import parse_beatmap

    overlap = overlap or OverlapConfig()
    range_cfg = range_cfg or GenerationRangeConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_dir = grid_dir or audio_grid_dir()

    set_dir = audio_path.parent
    audio_file = find_audio_file(set_dir)
    stem = resolve_grid_stem(audio_file, timing)
    try:
        grid, meta = load_audio_grid(grid_dir, stem)
    except FileNotFoundError:
        from audio2map.data.audio_grid import compute_audio_grid, save_audio_grid

        features, meta = compute_audio_grid(audio_file, timing)
        save_audio_grid(features, meta, grid_dir)
        grid = features

    audio_start, audio_end = audio_bar_range_from_duration(meta.duration_ms, timing)
    if reference_notes is None:
        try:
            reference_notes = parse_beatmap(audio_path).notes
        except Exception:
            reference_notes = None
    chart_start, chart_end = (
        chart_event_bar_range(reference_notes, timing) if reference_notes else (None, None)
    )
    gen_start, gen_end = resolve_generation_bar_range(
        audio_start_bar=audio_start,
        audio_end_bar=audio_end,
        chart_start_bar=chart_start,
        chart_end_bar=chart_end,
        cfg=range_cfg,
    )

    if single_window:
        start = single_window_start_bar if single_window_start_bar is not None else gen_start
        win_end = min(start + overlap.window_bars, gen_end)
        windows = [(start, win_end, start, win_end, 0)]
    else:
        windows = overlap_inference_bar_windows(gen_start, gen_end, overlap)

    report = GenerationReport(
        audio_start_bar=audio_start,
        audio_end_bar=audio_end,
        gen_start_bar=gen_start,
        gen_end_bar=gen_end,
        chart_start_bar=chart_start,
        chart_end_bar=chart_end,
        range_mode=range_cfg.mode,
        audio_pooling=model.audio_pooling,
    )

    model = model.to(device)
    cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=device)
    all_notes: list[ManiaNote] = []
    id_to_token = invert_vocab()

    for win_start, win_end, keep_start, keep_end, ctx_used in windows:
        window_bars = win_end - win_start
        audio_slice, _ = slice_audio_window(grid, meta, win_start, win_end)
        report.audio_ticks_per_window = audio_slice.shape[0]
        report.encoder_prefix_len = audio_slice.shape[0]

        win_start_tick = win_start * TICKS_PER_BAR
        initial = initial_row_at_tick(all_notes, timing, win_start_tick)
        token_ids = generate_window_tokens(
            model,
            audio=torch.tensor(audio_slice, dtype=torch.float32),
            cond_vec=cond_t,
            initial_row=initial,
            window_bars=window_bars,
            device=device,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        tokens = [id_to_token[i] for i in token_ids]
        if tokens[0] != TOKEN_BOS or tokens[-1] != TOKEN_EOS:
            continue
        win_notes = tokens_to_notes(tokens, timing, start_bar=win_start)
        keep_start_tick = keep_start * TICKS_PER_BAR
        keep_end_tick = keep_end * TICKS_PER_BAR
        kept = filter_notes_by_tick_range(
            win_notes, timing, keep_start_tick=keep_start_tick, keep_end_tick=keep_end_tick
        )
        all_notes.extend(kept)
        is_first = win_start == gen_start and ctx_used == 0
        report.windows.append(
            WindowGenerationLog(
                win_start=win_start,
                win_end=win_end,
                keep_start=keep_start,
                keep_end=keep_end,
                context_bars=ctx_used,
                is_first_window=is_first,
                notes_kept=len(kept),
            )
        )
        log.info(
            "window win=[%d,%d) keep=[%d,%d) ctx=%d first=%s kept=%d",
            win_start,
            win_end,
            keep_start,
            keep_end,
            ctx_used,
            is_first,
            len(kept),
        )

    notes = dedupe_notes(all_notes)
    first_b, last_b = _note_bar_range(notes, timing)
    report.first_note_bar = first_b
    report.last_note_bar = last_b
    if chart_end is not None and last_b is not None:
        report.tail_note_count = sum(
            1
            for n in notes
            if split_tick(ms_to_tick(n.time_ms, timing))[0] > chart_end
        )

    if report.windows:
        w0 = report.windows[0]
        log.info(
            "generation summary: audio=[%d,%d) gen=[%d,%d) first_keep_start=%d first_note_bar=%s "
            "encoder_prefix_len=%d pooling=%s",
            audio_start,
            audio_end,
            gen_start,
            gen_end,
            w0.keep_start,
            first_b,
            report.encoder_prefix_len,
            model.audio_pooling,
        )

    if return_report:
        return notes, report
    return notes
