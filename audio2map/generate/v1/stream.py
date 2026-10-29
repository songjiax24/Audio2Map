"""Segmented encoder context. A kept frame matches the full-span encode.

Mel is computed once for the coverage. Each encoder layer reads
``[e-512, e+512)``. Eight layers therefore need 4096 frames of real audio
on each side. The halo is clipped at the song, which is the same edge the
full encode sees. A segment slices that mel, then runs the encoder. RoPE
stays on the global frame index. Phase, log BPM, and ``u`` are pointwise
and use the kept frames only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch

from audio2map.features.audio.v1.constants import DT_MS
from audio2map.features.audio.v1.mel import frame_index, mel_span
from audio2map.grid.v1.beatgrid import BeatGrid
from audio2map.grid.v1.timefeat import phase_rows
from audio2map.model.v1.constants import D_MODEL, ENC_RADIUS, N_LAYERS
from audio2map.model.v1.model import V1ChartModel, phase_fourier
from audio2map.osu.v1.timing import Redline

# One radius per encoder layer. The right edge is exclusive, so this covers it.
ENCODER_HALO = N_LAYERS * ENC_RADIUS
# A shorter song is already one segment. Rate-1 frames are p50 11148, p90 21534,
# p99 34967. keep 8192 splits about three quarters of the 15383 charts.
# 65536 is one segment through p99.5 (38947 frames); 19 charts still split.
KEEP_FRAMES = 65536


def iter_segments(
    n_frames: int,
    *,
    keep: int = KEEP_FRAMES,
    halo: int = ENCODER_HALO,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(encode_lo, encode_hi, keep_lo, keep_hi)`` over ``n_frames``."""
    if n_frames <= 0:
        return
    if keep <= 0 or halo < 0:
        raise ValueError(f"keep={keep} halo={halo}")
    start = 0
    while start < n_frames:
        keep_lo = start
        keep_hi = min(n_frames, start + keep)
        encode_lo = max(0, keep_lo - halo)
        encode_hi = min(n_frames, keep_hi + halo)
        yield encode_lo, encode_hi, keep_lo, keep_hi
        start = keep_hi


def encode_context(
    model: V1ChartModel,
    pcm: np.ndarray,
    grid: BeatGrid,
    redlines: Sequence[Redline],
    *,
    device: torch.device,
    keep_frames: int = KEEP_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Context ``(1, S, D)``, global ``g`` ``(1, S)``, and how many segments ran.

    ``g`` stays int64. The context is encoder memory plus the 20-d phase.
    """
    if model.training:
        raise RuntimeError("segment encode is inference only")
    frame_lo = frame_index(grid.coverage_start_ms)
    frame_hi = frame_index(grid.coverage_end_ms)
    n_frames = frame_hi - frame_lo
    if n_frames <= 0:
        raise ValueError("coverage has no stem frames")
    span = mel_span(pcm, frame_lo * DT_MS, frame_hi * DT_MS)
    if span.frame_lo != frame_lo or span.frame_hi != frame_hi:
        raise RuntimeError(f"mel frames {span.frame_lo}:{span.frame_hi} != {frame_lo}:{frame_hi}")
    if span.mel.shape[1] != 2 * n_frames + 3:
        raise RuntimeError(f"mel width {span.mel.shape[1]} != {2 * n_frames + 3}")
    # Filled after the first window returns, so a one-segment song does not
    # hold this buffer under the encoder activation peak.
    kept: torch.Tensor | None = None
    segments = 0
    use_amp = device.type == "cuda"
    for encode_lo, encode_hi, keep_lo, keep_hi in iter_segments(n_frames, keep=keep_frames):
        segments += 1
        width = encode_hi - encode_lo
        mel_np = span.mel[:, 2 * encode_lo : 2 * encode_lo + 2 * width + 3]
        if mel_np.shape[1] != 2 * width + 3:
            raise RuntimeError(f"mel slice {mel_np.shape[1]} != {2 * width + 3}")
        mel = torch.from_numpy(np.ascontiguousarray(mel_np)).unsqueeze(0).to(device)
        origin = torch.tensor([frame_lo + encode_lo], dtype=torch.long, device=device)
        mask = torch.ones(1, width, dtype=torch.bool, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            encoded = model.encoder(mel, origin, mask)
        if kept is None:
            kept = torch.empty(1, n_frames, D_MODEL, device=device, dtype=torch.float32)
        local_lo = keep_lo - encode_lo
        local_hi = keep_hi - encode_lo
        # Residual adds stay fp32, so `.float()` is a view of the whole window
        # and would pin every halo. copy_ stores only this keep.
        kept[:, keep_lo:keep_hi, :].copy_(encoded[:, local_lo:local_hi, :])
        del encoded, mel, origin, mask
    if kept is None:
        raise RuntimeError("segment encode produced no context")
    memory = kept
    if memory.shape[1] != n_frames or memory.shape[-1] != D_MODEL:
        raise RuntimeError(f"segment context {tuple(memory.shape)} != {(1, n_frames, D_MODEL)}")
    phi, beat_b, log_bpm, rel_u = phase_rows(grid, redlines, frame_lo, n_frames)
    features = np.stack((phi, beat_b, log_bpm, rel_u), axis=0).astype(np.float32)
    stacked = torch.from_numpy(features).unsqueeze(0).to(device=device, dtype=memory.dtype)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        phase = phase_fourier(stacked[:, 0], stacked[:, 1], stacked[:, 2], stacked[:, 3])
        context = memory + model.phase(phase)
    g = torch.arange(frame_lo, frame_hi, device=device, dtype=torch.long).unsqueeze(0)
    return context, g, segments
