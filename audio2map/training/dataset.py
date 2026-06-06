"""PyTorch Dataset for v2 continuous-window training."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from audio2map.data.chart_bundle import (
    ChartBundle,
    build_sample_from_bundle,
    preload_chart_bundles,
)
from audio2map.data.eligible_charts import list_eligible_osu_paths
from audio2map.data.v2_dataset import V2TrainingSample
from audio2map.data.window_sampler import WindowSamplingConfig
from audio2map.osu.row_tokens import TOKEN_PAD, build_vocab
from audio2map.utils.paths import audio_grid_dir, chart_meta_dir, raw_dir

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

_MAX_CHART_ATTEMPTS = 32
_MAX_WINDOW_ATTEMPTS = 8
_MAX_INDEX_FALLBACKS = 64


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    samples_per_chart: int = 1
    build_grid_if_missing: bool = True
    require_grid: bool = False
    lazy: bool = False
    meta_manifest_path: Path | None = None

    def to_window_cfg(self) -> WindowSamplingConfig:
        return WindowSamplingConfig()


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
        self._warned_paths: set[Path] = set()
        manifest_path = self.cfg.meta_manifest_path or (chart_meta_dir() / "manifest.jsonl")
        self._meta_manifest = None
        if manifest_path.is_file():
            from audio2map.difficulty.chart_meta_manifest import load_chart_meta_manifest

            self._meta_manifest = load_chart_meta_manifest(manifest_path)

        if osu_paths is None:
            osu_paths = list_eligible_osu_paths(raw_root or raw_dir())
        requested = len(osu_paths) if osu_paths else 0

        if bundles is not None:
            self.bundles = bundles
            self.osu_paths = [b.path for b in bundles]
            self.lazy = False
        elif self.cfg.lazy:
            self.bundles = None
            self.lazy = True
            self.osu_paths = list(osu_paths)
            self._bundle_cache: dict[Path, ChartBundle] = {}
            from audio2map.data.chart_bundle import SharedGridCache

            self._grid_cache = SharedGridCache(self.grid_dir)
            self._cache_max = min(512, max(64, len(self.osu_paths)))
            if not self.osu_paths:
                raise ValueError("no chart paths for lazy dataset")
        else:
            input_paths = list(osu_paths)
            self.bundles = preload_chart_bundles(
                input_paths,
                grid_dir=self.grid_dir,
                require_grid=self.cfg.require_grid and not self.cfg.build_grid_if_missing,
                meta_manifest_path=manifest_path if manifest_path.is_file() else None,
            )
            self.osu_paths = [b.path for b in self.bundles]
            self.lazy = False
            skipped = requested - len(self.bundles)
            if skipped:
                logger.warning(
                    "preload skipped %d / %d charts (cond_vec, grid, or parse failures)",
                    skipped,
                    requested,
                )
            if not self.bundles:
                raise ValueError("no chart bundles available for dataset")

    def _num_charts(self) -> int:
        return len(self.osu_paths)

    def __len__(self) -> int:
        return self._num_charts() * self.cfg.samples_per_chart

    def _require_grid(self) -> bool:
        return self.cfg.require_grid and not self.cfg.build_grid_if_missing

    def _warn_once(self, path: Path, message: str, *args: object) -> None:
        if path in self._warned_paths:
            return
        self._warned_paths.add(path)
        logger.warning(message, path, *args)

    def _bundle_for_index(self, chart_index: int) -> ChartBundle | None:
        chart_index %= self._num_charts()
        if self.lazy:
            from audio2map.data.chart_bundle import preload_chart_bundle

            path = self.osu_paths[chart_index]
            cached = self._bundle_cache.get(path)
            if cached is not None:
                return cached
            bundle = preload_chart_bundle(
                path,
                grid_dir=self.grid_dir,
                require_grid=self._require_grid(),
                grid_cache=self._grid_cache,
                meta_manifest=self._meta_manifest,
            )
            if bundle is None:
                self._warn_once(path, "skip chart %s during lazy load")
                return None
            if len(self._bundle_cache) >= self._cache_max:
                self._bundle_cache.pop(next(iter(self._bundle_cache)))
            self._bundle_cache[path] = bundle
            return bundle
        return self.bundles[chart_index]

    def _any_loaded_bundle(self) -> ChartBundle | None:
        if self.lazy:
            if self._bundle_cache:
                return next(iter(self._bundle_cache.values()))
            return None
        if self.bundles:
            return self.bundles[0]
        return None

    @staticmethod
    def _sample_to_tensors(sample: V2TrainingSample) -> dict[str, torch.Tensor]:
        return {
            "token_ids": torch.tensor(sample.token_ids, dtype=torch.long),
            "loss_mask": torch.tensor(sample.loss_mask, dtype=torch.float32),
            "cond_vec": torch.tensor(sample.cond_vec, dtype=torch.float32),
            "audio": torch.from_numpy(sample.audio_features),
        }

    def _try_sample_bundle(
        self,
        bundle: ChartBundle,
        *,
        dataset_index: int,
        chart_offset: int,
    ) -> dict[str, torch.Tensor] | None:
        for attempt in range(_MAX_WINDOW_ATTEMPTS):
            if self.fixed_start_bar is None:
                rng = random.Random()
            else:
                rng = random.Random(self.seed + dataset_index + chart_offset + attempt)
            sample = build_sample_from_bundle(
                bundle,
                cfg=self.window_cfg,
                rng=rng,
                start_bar=self.fixed_start_bar,
            )
            if sample is not None:
                return self._sample_to_tensors(sample)
        logger.warning(
            "window sampling failed for %s (dataset_index=%s chart_offset=%s)",
            bundle.path,
            dataset_index,
            chart_offset,
        )
        return None

    def _getitem_from_charts(self, index: int) -> dict[str, torch.Tensor] | None:
        n = self._num_charts()
        attempts = min(_MAX_CHART_ATTEMPTS, n)
        for chart_offset in range(attempts):
            bundle = self._bundle_for_index((index + chart_offset) % n)
            if bundle is None:
                continue
            tensors = self._try_sample_bundle(
                bundle,
                dataset_index=index,
                chart_offset=chart_offset,
            )
            if tensors is not None:
                return tensors

        fallback = self._any_loaded_bundle()
        if fallback is not None:
            logger.warning("dataset index %s: falling back to chart %s", index, fallback.path)
            return self._try_sample_bundle(
                fallback,
                dataset_index=index,
                chart_offset=0,
            )
        return None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tensors = self._getitem_from_charts(index)
        if tensors is not None:
            return tensors

        for delta in range(1, min(len(self), _MAX_INDEX_FALLBACKS)):
            alt = (index + delta) % len(self)
            tensors = self._getitem_from_charts(alt)
            if tensors is not None:
                logger.warning(
                    "dataset index %s unavailable; served sample from index %s instead",
                    index,
                    alt,
                )
                return tensors

        raise ValueError(
            f"could not build training sample for dataset index {index} "
            f"(no loadable charts or windows in pool)"
        )

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        from audio2map.training.collate import pad_batch

        return pad_batch(batch)
