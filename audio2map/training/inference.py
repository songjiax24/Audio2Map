"""Overlap-window inference: slide fixed-size audio windows and stitch chart notes."""

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
from audio2map.osu.row_tokens import (
    TOKEN_BAR,
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
from audio2map.osu.schema import ManiaNote
from audio2map.training.decode import ChartDecodeState, bos_initial_prefix
from audio2map.training.config import MAX_DECODER_LEN
from audio2map.training.model import AudioChartModel
from audio2map.utils.paths import audio_grid_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OverlapConfig:
    """``context + keep + future = window_bars`` (default 8+4+4=16)."""

    window_bars: int = 16
    context_bars: int = 8
    keep_bars: int = 4
    future_bars: int = 4
    max_seq_len: int = MAX_DECODER_LEN

    def __post_init__(self) -> None:
        if self.context_bars + self.keep_bars + self.future_bars != self.window_bars:
            raise ValueError(
                f"context+keep+future must equal window_bars "
                f"({self.context_bars}+{self.keep_bars}+{self.future_bars}!={self.window_bars})"
            )


@dataclass(frozen=True, slots=True)
class GenerationRangeConfig:
    """``audio_full`` = whole mp3; ``reference_chart`` = clip to chart event bars."""

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
    """Resolve ``[gen_start_bar, gen_end_bar)`` from audio/chart bounds and ``mode``."""
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
    if gen_start_bar >= gen_end_bar:
        return []

    if gen_end_bar - gen_start_bar <= cfg.window_bars:
        s = gen_start_bar
        return [(s, s + cfg.window_bars, s, gen_end_bar, 0)]

    windows: list[tuple[int, int, int, int, int]] = []
    commit = gen_start_bar
    first = True

    while commit < gen_end_bar:
        if first:
            keep_start = commit
            keep_end = commit + cfg.context_bars
            win_start = gen_start_bar
            win_end = win_start + cfg.window_bars
            windows.append((win_start, win_end, keep_start, keep_end, 0))
            commit = keep_end
            first = False
        else:
            keep_start = commit
            win_start = keep_start - cfg.context_bars
            win_end = win_start + cfg.window_bars

            if win_end <= gen_end_bar:
                keep_end = keep_start + cfg.keep_bars
            else:
                win_end = gen_end_bar
                win_start = gen_end_bar - cfg.window_bars
                keep_end = gen_end_bar

            windows.append((win_start, win_end, keep_start, keep_end, cfg.context_bars))
            commit = keep_end

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


def initial_row_from_committed_bars(
    committed_bars: dict[int, list[int]],
    *,
    gen_start: int,
    win_start: int,
    vocab: dict[str, int],
):
    """Replay committed chart tokens before ``win_start`` to get hold state at window open."""
    if win_start <= gen_start:
        return (0, 0, 0, 0)

    decode = ChartDecodeState.from_initial_row(
        (0, 0, 0, 0),
        window_bars=max(1, win_start - gen_start),
        vocab=vocab,
    )
    for bar in range(gen_start, win_start):
        for tid in committed_bars.get(bar, []):
            decode.observe_and_advance_bar_if_needed(tid)
    return initial_row_from_active(decode.active_hold)


def extract_bar_token_ids(
    token_ids: list[int],
    id_to_token: dict[int, str],
    *,
    win_start_bar: int,
    bar: int,
) -> list[int]:
    """Token ids for a single absolute bar from one window generation."""
    tokens = [id_to_token[i] for i in token_ids]
    current_bar = win_start_bar - 1
    i = 2
    while i < len(tokens) and tokens[i] != TOKEN_EOS:
        if tokens[i] != TOKEN_BAR:
            i += 1
            continue
        current_bar += 1
        if current_bar != bar:
            i += 1
            continue
        out = [token_ids[i]]
        i += 1
        while i < len(tokens) and tokens[i] not in (TOKEN_BAR, TOKEN_EOS):
            out.append(token_ids[i])
            i += 1
        return out
    return []


def extract_bar_range_token_ids(
    token_ids: list[int],
    id_to_token: dict[int, str],
    *,
    win_start_bar: int,
    range_start_bar: int,
    range_end_bar: int,
) -> list[int]:
    out: list[int] = []
    for bar in range(range_start_bar, range_end_bar):
        out.extend(
            extract_bar_token_ids(
                token_ids,
                id_to_token,
                win_start_bar=win_start_bar,
                bar=bar,
            )
        )
    return out


def build_context_prompt_token_ids(
    *,
    committed_bars: dict[int, list[int]],
    win_start: int,
    keep_start: int,
    initial_row,
    vocab: dict[str, int],
) -> list[int] | None:
    """``<BOS> + <ROW_initial> + committed keep tokens for [win_start, keep_start)``."""
    if keep_start <= win_start:
        return None

    from audio2map.osu.row_tokens import TOKEN_BAR, row_state_to_token

    body: list[int] = []
    for bar in range(win_start, keep_start):
        if bar in committed_bars:
            body.extend(committed_bars[bar])
        else:
            log.warning(
                "missing committed tokens for context bar %d (win_start=%d keep_start=%d)",
                bar,
                win_start,
                keep_start,
            )
            body.append(vocab[TOKEN_BAR])

    prefix = bos_initial_prefix(vocab[row_state_to_token(initial_row)], vocab=vocab)
    return prefix + body


def commit_keep_bar_tokens(
    committed_bars: dict[int, list[int]],
    *,
    token_ids: list[int],
    id_to_token: dict[int, str],
    win_start: int,
    keep_start: int,
    keep_end: int,
) -> None:
    for bar in range(keep_start, keep_end):
        bar_ids = extract_bar_token_ids(
            token_ids, id_to_token, win_start_bar=win_start, bar=bar
        )
        if bar_ids:
            committed_bars[bar] = bar_ids


def assemble_chart_token_ids(
    committed_bars: dict[int, list[int]],
    *,
    gen_start: int,
    gen_end: int,
    vocab: dict[str, int],
) -> list[int]:
    """Merge per-bar keep tokens into one chart sequence for export."""
    from audio2map.osu.row_tokens import row_state_to_token

    initial = initial_row_from_committed_bars(
        committed_bars, gen_start=gen_start, win_start=gen_start, vocab=vocab
    )
    out = bos_initial_prefix(vocab[row_state_to_token(initial)], vocab=vocab)
    for bar in range(gen_start, gen_end):
        out.extend(committed_bars.get(bar, [vocab[TOKEN_BAR]]))
    out.append(vocab[TOKEN_EOS])
    return out


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
    max_seq_len: int | None = None,
    prompt_token_ids: list[int] | None = None,
) -> list[int]:
    """Generate one window token sequence (including BOS/EOS).

    When ``prompt_token_ids`` is set, it must be
    ``<BOS> + <ROW_initial> + context chart tokens`` for overlap windows.
    """
    from audio2map.osu.row_tokens import build_vocab, row_state_to_token

    cap = model.max_seq_len if max_seq_len is None else min(model.max_seq_len, max_seq_len)
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
    if prompt_token_ids is not None and len(token_ids) > 2:
        decode.require_bar_after_prompt()

    audio_b = audio.unsqueeze(0).to(device)
    cond_b = cond_vec.unsqueeze(0).to(device)

    while not decode.finished and len(token_ids) < cap:
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
    """Generate notes for a chart; default range is full mp3 (``audio_full``)."""
    from audio2map.audio.loader import find_audio_file
    from audio2map.osu.parser import parse_beatmap
    from audio2map.osu.row_tokens import build_vocab

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
    chart_start: int | None = None
    chart_end: int | None = None
    if reference_notes is not None:
        chart_start, chart_end = chart_event_bar_range(reference_notes, timing)
    elif range_cfg.mode == "reference_chart" or return_report:
        try:
            ref = parse_beatmap(audio_path).notes
            chart_start, chart_end = chart_event_bar_range(ref, timing)
        except Exception:
            pass
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
    id_to_token = invert_vocab()
    vocab = build_vocab()
    committed_bars: dict[int, list[int]] = {}

    for win_start, win_end, keep_start, keep_end, ctx_used in windows:
        window_bars = win_end - win_start
        audio_slice, _ = slice_audio_window(grid, meta, win_start, win_end)
        report.audio_ticks_per_window = audio_slice.shape[0]
        report.encoder_prefix_len = audio_slice.shape[0]

        initial = initial_row_from_committed_bars(
            committed_bars, gen_start=gen_start, win_start=win_start, vocab=vocab
        )
        prompt_ids = (
            build_context_prompt_token_ids(
                committed_bars=committed_bars,
                win_start=win_start,
                keep_start=keep_start,
                initial_row=initial,
                vocab=vocab,
            )
            if ctx_used > 0
            else None
        )
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
            max_seq_len=overlap.max_seq_len,
            prompt_token_ids=prompt_ids,
        )
        tokens = [id_to_token[i] for i in token_ids]
        if tokens[0] != TOKEN_BOS or tokens[-1] != TOKEN_EOS:
            log.warning(
                "skip window win=[%d,%d) keep=[%d,%d): invalid token boundaries "
                "(first=%r last=%r len=%d)",
                win_start,
                win_end,
                keep_start,
                keep_end,
                tokens[0] if tokens else None,
                tokens[-1] if tokens else None,
                len(tokens),
            )
            continue
        commit_keep_bar_tokens(
            committed_bars,
            token_ids=token_ids,
            id_to_token=id_to_token,
            win_start=win_start,
            keep_start=keep_start,
            keep_end=keep_end,
        )
        is_first = ctx_used == 0
        bars_kept = keep_end - keep_start
        report.windows.append(
            WindowGenerationLog(
                win_start=win_start,
                win_end=win_end,
                keep_start=keep_start,
                keep_end=keep_end,
                context_bars=ctx_used,
                is_first_window=is_first,
                notes_kept=bars_kept,
            )
        )
        log.info(
            "window win=[%d,%d) keep=[%d,%d) ctx=%d first=%s bars_kept=%d",
            win_start,
            win_end,
            keep_start,
            keep_end,
            ctx_used,
            is_first,
            bars_kept,
        )

    chart_token_ids = assemble_chart_token_ids(
        committed_bars, gen_start=gen_start, gen_end=gen_end, vocab=vocab
    )
    chart_tokens = [id_to_token[i] for i in chart_token_ids]
    notes = dedupe_notes(tokens_to_notes(chart_tokens, timing, start_bar=gen_start))
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
