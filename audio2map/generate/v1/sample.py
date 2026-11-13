"""Autoregressive sampling with an optional caller-supplied token prefix.

``legal_tokens`` is the same cursor used by teacher forcing. The decoder
step is ``cached_logits``. Forced tokens replay that step to build the same
cache and cursor; only the continuation is sampled. The default starts at BOS.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.generate.v1.trace import SamplingStep, StepObserver
from audio2map.model.v1.decode import CrossWindow
from audio2map.model.v1.kv import SelfKV, WindowStep, cached_logits
from audio2map.model.v1.model import V1ChartModel
from audio2map.tokens.errors import IllegalTokenError
from audio2map.tokens.v1.cursor_time import state_time_ms
from audio2map.tokens.v1.state import initial_cursor, legal_tokens, step
from audio2map.tokens.v1.vocab import TOKEN_BOS, build_vocab, invert_vocab

logger = logging.getLogger(__name__)

VALID = "valid"
ILLEGAL_TOKEN = "illegal_token"
TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class SampleResult:
    tokens: tuple[str, ...]
    validity: str
    position: int | None = None
    illegal_token: str | None = None


def sample_chart(
    model: V1ChartModel,
    context: torch.Tensor,
    g: torch.Tensor,
    grid: BeatGrid,
    *,
    cond: torch.Tensor,
    seed: int,
    temperature: float,
    prefix: Sequence[str] | None = None,
    history_start: int = 0,
    observer: StepObserver | None = None,
) -> list[str]:
    """Return prefix plus sampled continuation through EOS.

    ``prefix`` includes BOS and must be legal on this grid. Replaying it uses
    this call's condition, preserves open holds and absolute token positions,
    and consumes no sampling RNG. ``cond`` is the decoder MLP output.

    ``history_start`` indexes the first prefix token fed into a fresh KV cache.
    Earlier tokens only restore cursor/hold state and remain in the output.
    Absolute audio and token positions are retained; earlier hidden states are absent.
    """
    result = _sample(
        model,
        context,
        g,
        grid,
        cond=cond,
        seed=seed,
        temperature=temperature,
        constrained=True,
        prefix=prefix,
        history_start=history_start,
        observer=observer,
    )
    if result.validity != VALID:
        raise RuntimeError(f"constrained sampling ended with {result.validity}")
    return list(result.tokens)


def sample_unconstrained(
    model: V1ChartModel,
    context: torch.Tensor,
    g: torch.Tensor,
    grid: BeatGrid,
    *,
    cond: torch.Tensor,
    seed: int,
    temperature: float,
    observer: StepObserver | None = None,
) -> SampleResult:
    """Draw from the full vocabulary; stop and report the first illegal token."""
    return _sample(
        model,
        context,
        g,
        grid,
        cond=cond,
        seed=seed,
        temperature=temperature,
        constrained=False,
        observer=observer,
    )


def _sample(
    model,
    context,
    g,
    grid,
    *,
    cond,
    seed,
    temperature,
    constrained: bool,
    prefix: Sequence[str] | None = None,
    history_start: int = 0,
    observer: StepObserver | None = None,
) -> SampleResult:
    if model.training:
        raise RuntimeError("sampling is inference only")
    forced = (TOKEN_BOS,) if prefix is None else tuple(prefix)
    _check_prefix(forced, grid)
    if not isinstance(history_start, int) or not 0 <= history_start < len(forced):
        raise ValueError("history_start must index a token in the supplied prefix")
    device = context.device
    vocab = build_vocab()
    names = invert_vocab(vocab)
    use_amp = device.type == "cuda"
    frame_lo = int(g[0, 0].item())
    if int(g.shape[1]) != int(context.shape[1]):
        raise RuntimeError(f"g length {g.shape[1]} != context {context.shape[1]}")
    window = CrossWindow(frame_lo, int(context.shape[1]))
    kv = SelfKV(len(model.decoder), position_offset=history_start)
    stepper = WindowStep(model, kv, window, cond) if use_amp else None
    cursor = initial_cursor()
    tokens = list(forced[: history_start + 1])
    for token in tokens[1:]:
        step(cursor, token, grid)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    limit = _token_limit(grid)
    position = 0
    while not cursor.finished:
        if len(tokens) - 1 >= limit:
            return SampleResult(tuple(tokens), TRUNCATED, position=len(tokens) - 1)
        if position % 256 == 0:
            logger.info("sample position=%s beat=%s", position, cursor.beat_id)
        token_id = torch.tensor([[vocab[tokens[-1]]]], dtype=torch.long, device=device)
        cursor_frames = _cursor_frames(cursor, grid)
        # Floor the fp32 cursor, matching window_slots, without a GPU sync.
        cpu_c = torch.tensor(cursor_frames, dtype=torch.float32)
        floor_c = int(torch.floor(cpu_c).item())
        c_t = cpu_c.view(1, 1).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            window.prepare(model, context, floor_c)
            if stepper is not None and stepper.covers(position):
                logits = stepper.run(token_id, position, c_t)
            else:
                logits = cached_logits(
                    model,
                    token_id,
                    position,
                    kv,
                    cond,
                    c_t,
                    window=window,
                )
        allowed = legal_tokens(cursor, grid)
        if not allowed:
            raise RuntimeError("no legal token")
        choices = allowed if constrained else set(vocab)
        forced_step = len(tokens) < len(forced)
        cpu_logits = logits[0, 0].float().cpu() if observer is not None or not forced_step else None
        chosen = (
            forced[len(tokens)]
            if forced_step
            else _choose(cpu_logits, choices, vocab, names, temperature, generator)
        )
        if observer is not None:
            assert cpu_logits is not None
            pending = cursor.pending
            observer(
                SamplingStep(
                    position=len(tokens) - 1,
                    input_token=tokens[-1],
                    chosen_token=chosen,
                    raw_logits=cpu_logits.detach().clone(),
                    allowed_tokens=frozenset(allowed),
                    beat_id=cursor.beat_id,
                    last_fraction=cursor.last_fraction,
                    pending_fraction=None if pending is None else (pending.numer, pending.denom),
                    cursor_time_ms=state_time_ms(cursor, grid),
                    active_lanes=tuple(cursor.active),
                    expect_row=cursor.expect_row,
                    forced=forced_step,
                )
            )
        if chosen not in allowed:
            return SampleResult(
                tuple(tokens + [chosen]),
                ILLEGAL_TOKEN,
                position=len(tokens) - 1,
                illegal_token=chosen,
            )
        step(cursor, chosen, grid, allowed=allowed)
        tokens.append(chosen)
        position += 1
    return SampleResult(tuple(tokens), VALID)


def _check_prefix(prefix: tuple[str, ...], grid: BeatGrid) -> None:
    if not prefix or prefix[0] != TOKEN_BOS:
        raise IllegalTokenError(prefix[0] if prefix else "")
    cursor = initial_cursor()
    for token in prefix[1:]:
        step(cursor, token, grid)


def _cursor_frames(cursor, grid: BeatGrid) -> float:
    return state_time_ms(cursor, grid) / DT_MS


def _token_limit(grid: BeatGrid) -> int:
    points = sum(len(beat_points) for beat_points in grid.points)
    return 1 + len(grid.beats) + 2 * points


def _choose(
    logits: torch.Tensor,
    allowed: set[str],
    vocab: dict[str, int],
    names: dict[int, str],
    temperature: float,
    generator: torch.Generator,
) -> str:
    probs = sampling_distribution(logits, allowed, vocab, temperature)
    if temperature <= 0.0:
        return names[int(torch.argmax(probs).item())]
    return names[int(torch.multinomial(probs, 1, generator=generator).item())]


def sampling_distribution(
    logits: torch.Tensor,
    allowed: set[str] | frozenset[str],
    vocab: dict[str, int],
    temperature: float,
) -> torch.Tensor:
    """The native legal-token policy; greedy decoding is represented as a point mass."""
    mask = torch.full_like(logits, float("-inf"))
    ids = [vocab[name] for name in allowed]
    index = torch.tensor(ids, dtype=torch.long)
    mask[index] = logits[index]
    if not torch.isfinite(mask[index]).all():
        raise RuntimeError("legal logits are not finite")
    if temperature <= 0.0:
        result = torch.zeros_like(mask)
        result[int(torch.argmax(mask).item())] = 1
        return result
    return torch.softmax(mask / temperature, dim=-1)
