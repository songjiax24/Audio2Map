"""Overlap window generation and full-chart inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audio2map.data.audio_grid import load_audio_grid, resolve_grid_stem, slice_audio_window
from audio2map.data.window_sampler import audio_bar_range_from_duration
from audio2map.osu.grid_config import TICKS_PER_BAR
from audio2map.osu.row_tokens import (
    TOKEN_BOS,
    TOKEN_EOS,
    CanonicalTiming,
    initial_row_from_active,
    invert_vocab,
    ms_to_tick,
)
from audio2map.osu.round_trip import tokens_to_notes
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.training.decode import ChartDecodeState, bos_initial_prefix
from audio2map.training.model import AudioChartModel
from audio2map.utils.paths import audio_grid_dir


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


def overlap_inference_bar_windows(
    audio_start_bar: int,
    audio_end_bar: int,
    cfg: OverlapConfig,
) -> list[tuple[int, int, int, int]]:
    """Return ``(win_start, win_end, keep_start, keep_end)`` bar ranges."""
    windows: list[tuple[int, int, int, int]] = []
    bar = audio_start_bar
    while bar < audio_end_bar:
        win_end = min(bar + cfg.window_bars, audio_end_bar)
        win_start = win_end - cfg.window_bars
        if win_start < audio_start_bar:
            win_start = audio_start_bar
        keep_start = win_start + cfg.context_bars
        keep_end = min(keep_start + cfg.keep_bars, win_end)
        if keep_end > keep_start:
            windows.append((win_start, win_end, keep_start, keep_end))
        if win_end >= audio_end_bar:
            break
        bar += cfg.keep_bars
    return windows


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
    temperature: float = 1.0,
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
        mask = torch.full_like(logits, float("-inf"))
        for tid in allowed:
            mask[tid] = 0.0
        logits = logits + mask
        if temperature <= 0:
            next_id = int(logits.argmax().item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

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
        key = (n.time_ms, x.col, n.note_type, n.end_time_ms)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


@torch.no_grad()
def generate_chart_notes(
    model: AudioChartModel,
    *,
    audio_path: Path,
    timing: CanonicalTiming,
    cond_vec: np.ndarray,
    grid_dir: Path | None = None,
    overlap: OverlapConfig | None = None,
    device: torch.device | None = None,
    temperature: float = 0.0,
) -> list[ManiaNote]:
    """Generate a full chart over the audio bar range with overlap stitching."""
    from audio2map.audio.loader import find_audio_file

    overlap = overlap or OverlapConfig()
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
    windows = overlap_inference_bar_windows(audio_start, audio_end, overlap)

    model = model.to(device)
    cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=device)
    all_notes: list[ManiaNote] = []
    id_to_token = invert_vocab()

    for win_start, win_end, keep_start, keep_end in windows:
        window_bars = win_end - win_start
        audio_slice, _ = slice_audio_window(grid, meta, win_start, win_end)
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

    return dedupe_notes(all_notes)
