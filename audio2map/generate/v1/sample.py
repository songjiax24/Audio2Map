"""Constrained autoregressive sampling. Gold tokens are not an input.

``legal_tokens`` is the same cursor used by teacher forcing. The decoder
step is ``cached_logits``. ``teacher_decode`` is a separate gold-prefix path.
"""

from __future__ import annotations

import logging

import torch

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.model.v1.decode import CrossWindow
from audio2map.model.v1.kv import SelfKV, WindowStep, cached_logits
from audio2map.model.v1.model import V1ChartModel
from audio2map.tokens.v1.cursor_time import state_time_ms
from audio2map.tokens.v1.state import initial_cursor, legal_tokens, step
from audio2map.tokens.v1.vocab import TOKEN_BOS, build_vocab, invert_vocab

logger = logging.getLogger(__name__)


def sample_chart(
    model: V1ChartModel,
    context: torch.Tensor,
    g: torch.Tensor,
    grid: BeatGrid,
    *,
    cond: torch.Tensor,
    seed: int,
    temperature: float,
) -> list[str]:
    """Tokens from ``<BOS>`` through ``<EOS>``. ``cond`` is the decoder MLP output."""
    if model.training:
        raise RuntimeError("sampling is inference only")
    device = context.device
    vocab = build_vocab()
    names = invert_vocab(vocab)
    use_amp = device.type == "cuda"
    frame_lo = int(g[0, 0].item())
    if int(g.shape[1]) != int(context.shape[1]):
        raise RuntimeError(f"g length {g.shape[1]} != context {context.shape[1]}")
    window = CrossWindow(frame_lo, int(context.shape[1]))
    kv = SelfKV(len(model.decoder))
    stepper = WindowStep(model, kv, window, cond) if use_amp else None
    cursor = initial_cursor()
    tokens = [TOKEN_BOS]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    limit = _token_limit(grid)
    position = 0
    while not cursor.finished:
        if position >= limit:
            raise RuntimeError(f"sampling passed {limit} tokens without <EOS>")
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
        chosen = _choose(logits[0, 0].float().cpu(), allowed, vocab, names, temperature, generator)
        step(cursor, chosen, grid, allowed=allowed)
        tokens.append(chosen)
        position += 1
    return tokens


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
    mask = torch.full_like(logits, float("-inf"))
    ids = [vocab[name] for name in allowed]
    index = torch.tensor(ids, dtype=torch.long)
    mask[index] = logits[index]
    if not torch.isfinite(mask[index]).all():
        raise RuntimeError("legal logits are not finite")
    if temperature <= 0.0:
        pick = int(torch.argmax(mask).item())
        return names[pick]
    probs = torch.softmax(mask / temperature, dim=-1)
    pick = int(torch.multinomial(probs, 1, generator=generator).item())
    return names[pick]
