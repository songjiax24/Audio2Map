"""v0 generation adapter.

This is the only eval backend that calls overlap decoding and audio grids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audio2map.generate.overlap import (
    DecodeConfig,
    GenerationRangeConfig,
    OverlapConfig,
    generate_chart_notes,
)
from audio2map.generate.service import resolve_audio_file
from audio2map.grid import CanonicalTiming
from audio2map.model.model import AudioChartModel
from audio2map.osu.schema import ManiaNote

BACKEND_ID = "v0"


def generate_v0(
    model: AudioChartModel,
    *,
    path: Path,
    timing: CanonicalTiming,
    cond_vec: np.ndarray,
    device: torch.device,
    decode: DecodeConfig,
    constrain: bool,
    grid_dir: Path | None,
    build_grid_if_missing: bool,
    range_mode: str,
    post_margin_bars: int,
    reference_notes: list[ManiaNote],
    overlap: OverlapConfig | None = None,
    pre_margin_bars: int = 0,
) -> tuple[list[ManiaNote], str, list[dict]]:
    """Return notes, validity, and decode issues.

    Constrained success is ``valid``. That string is not withheld when
    ``missing_hold_tail`` is present. Constrained illegal or truncated windows
    still raise from ``generate_chart_notes``.
    """
    notes, report = generate_chart_notes(
        model,
        audio_path=resolve_audio_file(path),
        timing=timing,
        cond_vec=cond_vec,
        grid_dir=grid_dir,
        overlap=overlap or OverlapConfig(),
        range_cfg=GenerationRangeConfig(
            mode=range_mode,
            pre_margin_bars=pre_margin_bars,
            post_margin_bars=post_margin_bars,
        ),
        device=device,
        decode=decode,
        reference_notes=reference_notes,
        build_grid_if_missing=build_grid_if_missing,
        constrain=constrain,
    )
    validity = "valid" if constrain else str(report.chart_validity)
    return notes, validity, list(report.decode_issues)
