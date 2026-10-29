"""Training Flex cross-attention and the exact-window index it shares with decode.

``ca_train="flex"`` is the training default. ``"legacy"`` is the dense bool mask.
Q and KV are padded to a multiple of 128 so variable lengths that fall in the
same bucket reuse one compiled Flex graph. Sampling is unchanged. Each forward
records the real lengths and the buckets on the model.

The mask reads every query's own ``c_t``. It does not take the union of a chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from audio2map.model.v1.constants import CROSS_FUTURE, CROSS_PAST, D_MODEL
from audio2map.model.v1.windows import _as_long_index, _floor_index

CA_TRAIN_FLEX = "flex"
CA_TRAIN_LEGACY = "legacy"
BUCKET = 128
_WINDOW = CROSS_PAST + CROSS_FUTURE
# A 512-key window starts mid-block and then covers five blocks of 128.
_BLOCKS_PER_QUERY = (_WINDOW + BUCKET - 2) // BUCKET + 1
_PLAN: list[FlexPlan | None] = [None]
_COMPILED: list = []


def normalize_ca_train(mode: str) -> str:
    key = mode.strip().lower()
    if key not in (CA_TRAIN_FLEX, CA_TRAIN_LEGACY):
        raise ValueError(f"ca_train must be {CA_TRAIN_FLEX!r} or {CA_TRAIN_LEGACY!r}, got {mode!r}")
    return key


def bucket_length(length: int, multiple: int = BUCKET) -> int:
    """Next multiple of ``multiple``. Same bucket shares a compiled Flex shape."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    return ((length + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class CrossLog:
    """One forward. ``bucket_hit`` means this Q/KV bucket was already seen.

    ``fallback="cpu-dense"`` means Flex was requested but the tensors are on CPU.
    Flex has no CPU backward, so that forward uses the legacy dense mask.
    """

    impl: str
    compiled: bool
    q_len: int
    kv_len: int
    q_bucket: int
    kv_bucket: int
    bucket_hit: bool
    fallback: str = ""


def flex_compiled(reference: torch.Tensor) -> bool:
    """Compile only for CUDA fp16/bf16, including autocast that still leaves a fp32 residual."""
    if not reference.is_cuda:
        return False
    if reference.dtype in (torch.float16, torch.bfloat16):
        return True
    return bool(torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") in (torch.float16, torch.bfloat16))


@dataclass
class FlexPlan:
    """Shared by the 8 decoder layers of one forward."""

    block: BlockMask
    empty: torch.Tensor
    q_bucket: int
    kv_bucket: int
    compiled: bool
    floor_c: torch.Tensor
    g_pad: torch.Tensor
    visible: torch.Tensor
    q_keep: torch.Tensor
    empty_pad: torch.Tensor


def window_slots(
    c_t: torch.Tensor,
    g: torch.Tensor,
    audio_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query indices into the crop. Not the union of several queries.

    ``g`` is the integer ``frame_lo + arange(S)``. ``c_t`` is fp32.
    ``valid`` is ``[floor(c_t)-128, floor(c_t)+384)`` intersected with visible keys.
    """
    batch, _queries = c_t.shape
    keys = _as_long_index("g", g)
    floor_c = _floor_index("c_t", c_t)
    n_keys = keys.shape[1]
    origin = keys[:, :1]
    left = floor_c - CROSS_PAST - origin
    slots = torch.arange(_WINDOW, device=keys.device)
    index = left.unsqueeze(-1) + slots
    inside = (index >= 0) & (index < n_keys)
    safe = index.clamp(min=0, max=max(n_keys - 1, 0))
    visible = torch.gather(audio_mask, 1, safe.reshape(batch, -1)).view(batch, -1, _WINDOW)
    return safe, inside & visible


def _mask_mod(b_idx, h_idx, q_idx, kv_idx):
    del h_idx
    plan = _PLAN[0]
    assert plan is not None
    floor_c = plan.floor_c[b_idx, q_idx]
    key = plan.g_pad[b_idx, kv_idx]
    left = floor_c - CROSS_PAST
    right = floor_c + CROSS_FUTURE
    query_ok = plan.q_keep[b_idx, q_idx]
    window = (key >= left) & (key < right) & plan.visible[b_idx, kv_idx] & query_ok
    dummy = plan.empty_pad[b_idx, q_idx] & (kv_idx == 0)
    return window | dummy


def _compiled_flex():
    if not _COMPILED:
        torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), 32)
        _COMPILED.append(torch.compile(flex_attention))
    return _COMPILED[0]


def _ordered_block_index(grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Present blocks first, matching Flex's ``(kv_num_blocks, kv_indices)``."""
    dense = grid.to(dtype=torch.int32)
    count = dense.sum(dim=-1)
    index = torch.argsort(dense, dim=-1, descending=True, stable=True)
    return count.to(dtype=torch.int32), index.to(dtype=torch.int32)


def _block_mask(plan: FlexPlan, kv_len: int) -> BlockMask:
    """Tiles each query window meets. Every touched block stays partial.

    ``g`` is ``frame_lo + arange`` on the unpadded prefix, so each query is one
    index interval of at most 512 keys. Empty queries, including the bucket
    tail, also touch block 0. ``mask_mod`` drops holes, edges, and the dummy key.
    """
    floor_c = plan.floor_c
    origin = plan.g_pad[:, :1]
    visible = plan.visible[:, :kv_len]
    batch, queries = floor_c.shape
    prefix = visible.to(dtype=torch.int32).cumsum(dim=1)
    prefix = F.pad(prefix, (1, 0))
    lo = (floor_c - CROSS_PAST - origin).clamp(0, kv_len)
    hi = (floor_c + CROSS_FUTURE - origin).clamp(0, kv_len)
    keep = plan.q_keep
    blank = torch.zeros_like(lo)
    lo = torch.where(keep, lo, blank)
    hi = torch.where(keep, hi, blank)
    n_qb = plan.q_bucket // BUCKET
    n_kb = plan.kv_bucket // BUCKET
    grid = torch.zeros(batch, 1, n_qb, n_kb, dtype=torch.int32, device=floor_c.device)
    query_block = torch.arange(queries, device=floor_c.device) // BUCKET
    query_block = query_block.view(1, queries).expand(batch, queries)
    rows = torch.arange(batch, device=floor_c.device)[:, None].expand(batch, queries)
    for offset in range(_BLOCKS_PER_QUERY):
        block = lo // BUCKET + offset
        left = torch.maximum(lo, block * BUCKET)
        right = torch.minimum(hi, (block + 1) * BUCKET)
        seen = prefix.gather(1, right.clamp(0, kv_len))
        seen = seen - prefix.gather(1, left.clamp(0, kv_len))
        valid = keep & (lo < hi) & (right > left) & (seen > 0)
        valid = valid & (block >= 0) & (block < n_kb)
        grid[rows[valid], 0, query_block[valid], block[valid]] = 1
    empty = plan.empty_pad
    dummy = torch.zeros(batch, queries, dtype=torch.long, device=floor_c.device)
    grid[rows[empty], 0, query_block[empty], dummy[empty]] = 1
    count, index = _ordered_block_index(grid)
    return BlockMask.from_kv_blocks(
        count,
        index,
        None,
        None,
        BLOCK_SIZE=BUCKET,
        mask_mod=_mask_mod,
        seq_lengths=(plan.q_bucket, plan.kv_bucket),
    )


def _pad_tail(value: torch.Tensor, length: int, fill: float | bool | int) -> torch.Tensor:
    extra = length - value.shape[1]
    if extra <= 0:
        return value
    return F.pad(value, (0, extra), value=fill)


def prepare_flex(
    c_t: torch.Tensor,
    g: torch.Tensor,
    audio_mask: torch.Tensor,
    *,
    compiled: bool,
    bucket_hit: bool,
) -> tuple[FlexPlan, CrossLog]:
    """One BlockMask for every decoder layer. Empty queries get a dummy key.

    The block list is those 128-wide tiles. ``mask_mod`` is not evaluated on the
    full Q×KV grid.
    """
    _batch, q_len = c_t.shape
    kv_len = g.shape[1]
    q_bucket = bucket_length(q_len)
    kv_bucket = bucket_length(kv_len)
    keys = _as_long_index("g", g)
    floor_c = _floor_index("c_t", c_t)
    _index, valid = window_slots(floor_c, keys, audio_mask)
    empty = ~valid.any(dim=-1)
    q_keep = _pad_tail(torch.ones(c_t.shape[:2], dtype=torch.bool, device=c_t.device), q_bucket, False)
    empty_pad = _pad_tail(empty, q_bucket, True)
    plan = FlexPlan(
        block=None,  # type: ignore[arg-type]
        empty=empty,
        q_bucket=q_bucket,
        kv_bucket=kv_bucket,
        compiled=compiled,
        floor_c=_pad_tail(floor_c, q_bucket, 0),
        g_pad=_pad_tail(keys, kv_bucket, 0),
        visible=_pad_tail(audio_mask, kv_bucket, False),
        q_keep=q_keep,
        empty_pad=empty_pad,
    )
    _PLAN[0] = plan
    plan.block = _block_mask(plan, kv_len)
    log = CrossLog(
        impl=CA_TRAIN_FLEX,
        compiled=compiled,
        q_len=q_len,
        kv_len=kv_len,
        q_bucket=q_bucket,
        kv_bucket=kv_bucket,
        bucket_hit=bucket_hit,
    )
    return plan, log


def flex_cross(attn, query: torch.Tensor, memory: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor, plan: FlexPlan) -> torch.Tensor:
    """FlexAttention. A query with no real key is zeroed before ``out_proj``."""
    batch, q_len, _ = query.shape
    s_len = memory.shape[1]
    q = attn.q_proj(query).view(batch, q_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(memory).view(batch, s_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(memory).view(batch, s_len, attn.n_heads, attn.head_dim).transpose(1, 2)
    q = attn.rotary.apply(q, q_pos)
    k = attn.rotary.apply(k, k_pos)
    q_pad = plan.q_bucket - q_len
    k_pad = plan.kv_bucket - s_len
    if q_pad:
        q = F.pad(q, (0, 0, 0, q_pad))
    if k_pad:
        k = F.pad(k, (0, 0, 0, k_pad))
        v = F.pad(v, (0, 0, 0, k_pad))
    _PLAN[0] = plan
    runner = _compiled_flex() if plan.compiled else flex_attention
    out = runner(q, k, v, block_mask=plan.block)
    out = out[:, :, :q_len].transpose(1, 2).contiguous().view(batch, q_len, D_MODEL)
    out = out.masked_fill(plan.empty.unsqueeze(-1), 0.0)
    return attn.out_proj(out)
