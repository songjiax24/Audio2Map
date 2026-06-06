"""Window- and chart-level evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from audio2map.data.v2_dataset import build_training_sample
from audio2map.data.window_sampler import WindowSamplingConfig
from audio2map.eval.note_match import NoteMatchStats, compare_note_lists
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.round_trip import quantize_notes, round_trip_beatmap, tokens_to_notes
from audio2map.osu.row_tokens import CanonicalTiming, invert_vocab
from audio2map.osu.schema import Beatmap


@dataclass(slots=True)
class AggregateStats:
    charts: int = 0
    roundtrip_ok: int = 0
    note: NoteMatchStats = field(default_factory=NoteMatchStats)
    token_correct: int = 0
    token_total: int = 0
    split_token: object | None = None

    def merge_note(self, s: NoteMatchStats) -> None:
        self.note.tp += s.tp
        self.note.fp += s.fp
        self.note.fn += s.fn
        self.note.tap_tp += s.tap_tp
        self.note.tap_fp += s.tap_fp
        self.note.tap_fn += s.tap_fn
        self.note.hold_tp += s.hold_tp
        self.note.hold_fp += s.hold_fp
        self.note.hold_fn += s.hold_fn
        if s.ms_deltas:
            if self.note.ms_deltas is None:
                self.note.ms_deltas = []
            self.note.ms_deltas.extend(s.ms_deltas)

    def to_dict(self) -> dict:
        out = {
            "charts": self.charts,
            "roundtrip_rate": self.roundtrip_ok / self.charts if self.charts else 0.0,
            "notes": self.note.to_dict(),
            "token_acc": self.token_correct / self.token_total if self.token_total else 0.0,
        }
        if self.split_token is not None:
            out["split_token_acc"] = self.split_token.to_dict()
        return out


def eval_roundtrip(paths: list[Path]) -> AggregateStats:
    agg = AggregateStats()
    for path in paths:
        bm = parse_beatmap(path)
        ok = round_trip_beatmap(bm)
        agg.charts += 1
        agg.roundtrip_ok += int(ok)
    return agg


def eval_window_notes(
    paths: list[Path],
    *,
    seed: int = 0,
    build_grid_if_missing: bool = False,
) -> AggregateStats:
    """GT tokens → notes vs quantize(GT) for one random window per chart."""
    if build_grid_if_missing:
        return _eval_window_with_dataset(paths, seed=seed, build_grid_if_missing=True)
    return _eval_window_tokens_only(paths, seed=seed)


def _eval_window_tokens_only(paths: list[Path], *, seed: int) -> AggregateStats:
    """Fast window note eval without audio grid (encode/decode only)."""
    import random

    from audio2map.data.window_sampler import WindowSamplingConfig, sample_window_bars
    from audio2map.osu.row_tokens import beatmap_to_window_tokens

    agg = AggregateStats()
    rng = random.Random(seed)
    cfg = WindowSamplingConfig()

    for path in paths:
        bm = parse_beatmap(path)
        timing = CanonicalTiming.from_beatmap(bm)
        from audio2map.data.window_sampler import chart_bar_range, audio_bar_range_from_duration
        from audio2map.audio.loader import find_audio_file, audio_duration_ms, load_mono_audio

        chart_start, chart_end = chart_bar_range(bm, timing)
        y, sr = load_mono_audio(find_audio_file(path.parent))
        audio_start, audio_end = audio_bar_range_from_duration(audio_duration_ms(y, sr), timing)
        sampled = sample_window_bars(
            audio_start_bar=audio_start,
            audio_end_bar=audio_end,
            cfg=cfg,
            rng=rng,
        )
        if sampled is None:
            continue
        start, end = sampled
        tokens = beatmap_to_window_tokens(bm, start_bar=start, window_bars=end - start, timing=timing)
        exp_all = quantize_notes(bm.notes, timing)
        exp = _notes_in_bar_window(exp_all, timing, start, end)
        pred = tokens_to_notes(tokens, timing, start_bar=start)
        agg.charts += 1
        agg.merge_note(compare_note_lists(pred, exp, timing))
    return agg


def _eval_window_with_dataset(
    paths: list[Path],
    *,
    seed: int,
    build_grid_if_missing: bool,
) -> AggregateStats:
    import random

    agg = AggregateStats()
    rng = random.Random(seed)
    for path in paths:
        sample = build_training_sample(
            path,
            cfg=WindowSamplingConfig(),
            rng=rng,
            build_grid_if_missing=build_grid_if_missing,
        )
        if sample is None:
            continue
        bm = parse_beatmap(path)
        timing = CanonicalTiming.from_beatmap(bm)
        start = sample.window["start_bar"]
        end = start + sample.window["window_bars"]
        exp_all = quantize_notes(bm.notes, timing)
        exp = _notes_in_bar_window(exp_all, timing, start, end)
        pred = tokens_to_notes(sample.tokens, timing, start_bar=start)
        agg.charts += 1
        agg.merge_note(compare_note_lists(pred, exp, timing))
    return agg


def _notes_in_bar_window(
    notes: list,
    timing: CanonicalTiming,
    start_bar: int,
    end_bar: int,
) -> list:
    from audio2map.osu.grid_config import TICKS_PER_BAR
    from audio2map.osu.row_tokens import ms_to_tick

    lo = start_bar * TICKS_PER_BAR
    hi = end_bar * TICKS_PER_BAR
    out = []
    for n in notes:
        t = ms_to_tick(n.time_ms, timing)
        if lo <= t < hi:
            out.append(n)
    return out


def eval_teacher_forcing(
    model,
    paths: list[Path],
    *,
    device,
    seed: int = 0,
    build_grid_if_missing: bool = False,
    fixed_start_bar: int | None = None,
) -> AggregateStats:
    """Token accuracy with GT prefix (diagnostic upper bound)."""
    import random
    import torch

    from audio2map.data.v2_dataset import build_training_sample
    from audio2map.data.window_sampler import WindowSamplingConfig
    from audio2map.eval.token_accuracy import SplitTokenAccuracy, accumulate_split_accuracy

    agg = AggregateStats()
    agg.split_token = SplitTokenAccuracy()
    rng = random.Random(seed)
    id_to_tok = invert_vocab()
    model.eval()

    with torch.no_grad():
        for path in paths:
            sample = build_training_sample(
                path,
                cfg=WindowSamplingConfig(),
                rng=rng,
                build_grid_if_missing=build_grid_if_missing,
                start_bar=fixed_start_bar,
            )
            if sample is None:
                continue
            agg.charts += 1
            token_ids = torch.tensor([sample.token_ids], dtype=torch.long, device=device)
            loss_mask = torch.tensor([sample.loss_mask], dtype=torch.float32, device=device)
            audio = torch.tensor([sample.audio_features], dtype=torch.float32, device=device)
            cond = torch.tensor([sample.cond_vec], dtype=torch.float32, device=device)
            attn = torch.ones(1, token_ids.shape[1], dtype=torch.bool, device=device)

            logits = model(audio, cond, token_ids, attn_mask=attn)
            targets = token_ids[:, 1:]
            mask = loss_mask[:, 1:]
            pred = logits.argmax(dim=-1)
            correct = ((pred == targets) & mask.bool()).sum().item()
            total = mask.sum().item()
            agg.token_correct += int(correct)
            agg.token_total += int(total)
            agg.split_token.merge(accumulate_split_accuracy(pred[0], targets[0], mask[0]))

            timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
            rebuilt_ids = token_ids[0].clone()
            rebuilt_ids[1:] = pred[0]
            if rebuilt_ids[-1].item() != token_ids[0, -1].item():
                rebuilt_ids[-1] = token_ids[0, -1]
            pred_toks = [id_to_tok[int(i)] for i in rebuilt_ids.tolist()]
            start = sample.window["start_bar"]
            end = start + sample.window["window_bars"]
            pred_notes = tokens_to_notes(pred_toks, timing, start_bar=start)
            exp = _notes_in_bar_window(
                quantize_notes(parse_beatmap(path).notes, timing),
                timing,
                start,
                end,
            )
            agg.merge_note(compare_note_lists(pred_notes, exp, timing))

    return agg


def eval_inference_chart(
    model,
    path: Path,
    *,
    device,
    cond_vec: np.ndarray,
    overlap=None,
    temperature: float = 0.0,
    build_grid_if_missing: bool = True,
) -> NoteMatchStats:
    from audio2map.training.inference import OverlapConfig, generate_chart_notes

    timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
    expected = quantize_notes(parse_beatmap(path).notes, timing)
    pred = generate_chart_notes(
        model,
        audio_path=path,
        timing=timing,
        cond_vec=cond_vec,
        overlap=overlap or OverlapConfig(),
        device=device,
        temperature=temperature,
    )
    return compare_note_lists(pred, expected, timing)


def eval_generate_window(
    model,
    path: Path,
    *,
    device,
    start_bar: int,
    build_grid_if_missing: bool = True,
    temperature: float = 0.0,
) -> tuple[AggregateStats, NoteMatchStats]:
    """Autoregressive constrained decode on one fixed window."""
    import torch

    from audio2map.data.cond_vec import build_cond_vec
    from audio2map.data.v2_dataset import build_training_sample
    from audio2map.difficulty.chart_meta import compute_chart_meta
    from audio2map.osu.row_tokens import invert_vocab, row_state_from_token
    from audio2map.training.inference import generate_window_tokens

    sample = build_training_sample(
        path,
        start_bar=start_bar,
        build_grid_if_missing=build_grid_if_missing,
        chart_meta=compute_chart_meta(path),
    )
    if sample is None:
        raise RuntimeError(f"cannot build sample for {path}")

    timing = CanonicalTiming.from_beatmap(parse_beatmap(path))
    end = start_bar + sample.window["window_bars"]
    initial = row_state_from_token(sample.tokens[1])
    cond = torch.tensor(sample.cond_vec, dtype=torch.float32, device=device)

    token_ids = generate_window_tokens(
        model,
        audio=torch.tensor(sample.audio_features, dtype=torch.float32),
        cond_vec=cond,
        initial_row=initial,
        window_bars=sample.window["window_bars"],
        device=device,
        temperature=temperature,
    )
    pred_toks = [invert_vocab()[i] for i in token_ids]
    pred_notes = tokens_to_notes(pred_toks, timing, start_bar=start_bar)
    exp = _notes_in_bar_window(
        quantize_notes(parse_beatmap(path).notes, timing),
        timing,
        start_bar,
        end,
    )
    note_stats = compare_note_lists(pred_notes, exp, timing)

    agg = AggregateStats(charts=1)
    agg.token_correct = sum(
        int(p == t) for p, t in zip(token_ids[2:-1], sample.token_ids[2:-1], strict=False)
    )
    agg.token_total = len(sample.token_ids) - 3
    agg.merge_note(note_stats)
    return agg, note_stats
