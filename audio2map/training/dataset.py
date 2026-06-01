"""PyTorch Dataset for v2 continuous-window training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from audio2map.data.chart_bundle import (
    ChartBundle,
    build_sample_from_bundle,
    preload_chart_bundles,
)
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.data.window_sampler import WindowSamplingConfig
from audio2map.osu.row_tokens import TOKEN_PAD, build_vocab
from audio2map.utils.paths import audio_grid_dir, raw_dir

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    window_bars_choices: tuple[int, ...] = (8,)
    pre_event_margin_bars: int = 4
    post_event_margin_bars: int = 4
    samples_per_chart: int = 1
    build_grid_if_missing: bool = True
    require_grid: bool = False
    lazy: bool = False

    def to_window_cfg(self) -> WindowSamplingConfig:
        return WindowSamplingConfig(
            window_bars=self.window_bars_choices[0],
            window_bars_choices=self.window_bars_choices,
            pre_event_margin_bars=self.pre_event_margin_bars,
            post_event_margin_bars=self.post_event_margin_bars,
        )


class Audio2MapV2Dataset(Dataset):
    """On-the-fly v2 window samples from eligible ``*.osu`` paths."""

    def __init__(
        self,
        osu_paths: list[Path] | None = None,
        *,
        raw_root: Path | None = None,
        cfg: DatasetConfig | None = None,
        seed: int = 0,
        grid_dir: Path | None = None,
        fixed_start_bar: int | None = None,
        bundles: list[ChartBundle] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for Audio2MapV2Dataset")

        self.cfg = cfg or DatasetConfig()
        self.window_cfg = self.cfg.to_window_cfg()
        self.seed = seed
        self.grid_dir = grid_dir or audio_grid_dir()
        self.fixed_start_bar = fixed_start_bar
        self.vocab = build_vocab()
        self.pad_id = self.vocab[TOKEN_PAD]

        if osu_paths is None:
            osu_paths = list_eligible_osu_paths(raw_root or raw_dir())
        self.osu_paths = list(osu_paths)

        if bundles is not None:
            self.bundles = bundles
            self.lazy = False
        elif self.cfg.lazy:
            self.bundles = None
            self.lazy = True
            self._bundle_cache: dict[Path, ChartBundle] = {}
            self._cache_max = min(512, max(64, len(self.osu_paths)))
            if not self.osu_paths:
                raise ValueError("no chart paths for lazy dataset")
        else:
            self.bundles = preload_chart_bundles(
                self.osu_paths,
                grid_dir=self.grid_dir,
                require_grid=self.cfg.require_grid and not self.cfg.build_grid_if_missing,
            )
            self.lazy = False
            if not self.bundles:
                raise ValueError("no chart bundles available for dataset")

    def __len__(self) -> int:
        n = len(self.osu_paths) if self.lazy else len(self.bundles)
        return n * self.cfg.samples_per_chart

    def _bundle_for_index(self, index: int) -> ChartBundle:
        if self.lazy:
            from audio2map.data.chart_bundle import preload_chart_bundle

            path = self.osu_paths[index % len(self.osu_paths)]
            cached = self._bundle_cache.get(path)
            if cached is not None:
                return cached
            bundle = preload_chart_bundle(
                path,
                grid_dir=self.grid_dir,
                require_grid=self.cfg.require_grid and not self.cfg.build_grid_if_missing,
            )
            if bundle is None:
                raise RuntimeError(f"lazy load failed for {path}")
            if len(self._bundle_cache) >= self._cache_max:
                self._bundle_cache.pop(next(iter(self._bundle_cache)))
            self._bundle_cache[path] = bundle
            return bundle
        return self.bundles[index % len(self.bundles)]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        bundle = self._bundle_for_index(index)
        for attempt in range(8):
            rng = random.Random() if self.fixed_start_bar is None else random.Random(self.seed + index + attempt)
            sample = build_sample_from_bundle(
                bundle,
                cfg=self.window_cfg,
                rng=rng,
                start_bar=self.fixed_start_bar,
            )
            if sample is not None:
                return {
                    "token_ids": torch.tensor(sample.token_ids, dtype=torch.long),
                    "loss_mask": torch.tensor(sample.loss_mask, dtype=torch.float32),
                    "cond_vec": torch.tensor(sample.cond_vec, dtype=torch.float32),
                    "audio": torch.from_numpy(sample.audio_features),
                }
        raise RuntimeError(f"failed to build sample for {bundle.path}")

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        from audio2map.training.collate import pad_batch

        return pad_batch(batch)
