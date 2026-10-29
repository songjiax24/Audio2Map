"""A mel slice through the stem matches those frames of the full convolution."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from audio2map.features.audio.v1.mel import mel_span
from audio2map.model.audio_stem import AudioStem


def _conv(stem: AudioStem, mel: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(mel)).unsqueeze(0)
    weight = stem.conv1.weight
    tensor = tensor.to(device=weight.device)
    with torch.no_grad():
        return stem(tensor)


def test_mel_slice_matches_the_full_conv() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(22_050 * 4).astype(np.float32) * 0.1
    stem = AudioStem()
    full = mel_span(y, 0.0, 4000.0)
    frames = full.frame_hi - full.frame_lo
    assert full.mel.shape[1] == 2 * frames + 3
    lo, hi = 20, 80
    assert hi <= frames
    width = hi - lo
    part = full.mel[:, 2 * lo : 2 * lo + 2 * width + 3]
    assert part.shape[1] == 2 * width + 3
    whole = _conv(stem, full.mel)
    sliced = _conv(stem, part)
    assert int(whole.shape[1]) == frames
    assert int(sliced.shape[1]) == width
    gap = float((sliced - whole[:, lo:hi]).abs().max())
    assert gap < 1e-5, gap


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_stem_matches_the_cpu_stem() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(22_050).astype(np.float32) * 0.1
    mel = mel_span(y, 0.0, 800.0).mel
    cpu = AudioStem()
    cuda = AudioStem().to("cuda")
    cuda.load_state_dict(cpu.state_dict())
    cpu_rows = _conv(cpu, mel)
    gpu_rows = _conv(cuda, mel).cpu()
    gap = float((gpu_rows - cpu_rows).abs().max())
    assert gap < 1e-4, gap
