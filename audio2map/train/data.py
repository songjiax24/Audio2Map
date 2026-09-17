"""PyTorch Dataset + batch padding for continuous-window training."""

from __future__ import annotations

import logging
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from audio2map.dataset.bundle import (
    ChartBundle,
    SharedGridCache,
    grid_covers_training_window,
    preload_chart_bundle,
    preload_chart_bundles,
)
from audio2map.dataset.filter import list_eligible_osu_paths
from audio2map.dataset.sample import TrainingSample, build_sample
from audio2map.features.cond import load_chart_meta_manifest
from audio2map.model.config import WINDOW_BARS
from audio2map.tokens import TOKEN_PAD, build_vocab
from audio2map.utils.paths import audio_grid_dir, chart_meta_dir, raw_dir

logger = logging.getLogger(__name__)

_MAX_WINDOW_ATTEMPTS = 8
_PAD_ID = build_vocab()[TOKEN_PAD]


class AudioChartDataset(Dataset):
    """On-the-fly window samples from eligible ``*.osu`` paths."""

    def __init__(
        self,
        osu_paths: list[Path] | None = None,
        *,
        raw_root: Path | None = None,
        seed: int = 0,
        grid_dir: Path | None = None,
        fixed_start_bar: int | None = None,
        samples_per_chart: int = 1,
        build_grid_if_missing: bool = True,
        lazy: bool = False,
        meta_manifest_path: Path | None = None,
    ) -> None:
        self.seed = seed
        self.grid_dir = grid_dir or audio_grid_dir()
        self.fixed_start_bar = fixed_start_bar
        self.samples_per_chart = samples_per_chart
        self._build_grid_if_missing = build_grid_if_missing
        manifest_path = meta_manifest_path or (chart_meta_dir() / "manifest.jsonl")

        if osu_paths is None:
            osu_paths = list_eligible_osu_paths(raw_root or raw_dir())
        requested = len(osu_paths) if osu_paths else 0

        if lazy:
            self.bundles = None
            self.lazy = True
            self.osu_paths = list(osu_paths)
            if not build_grid_if_missing:
                kept = [
                    p
                    for p in self.osu_paths
                    if grid_covers_training_window(p, grid_dir=self.grid_dir)
                ]
                dropped = len(self.osu_paths) - len(kept)
                if dropped:
                    logger.warning(
                        "dropped %d / %d lazy charts shorter than %d bars",
                        dropped,
                        len(self.osu_paths),
                        WINDOW_BARS,
                    )
                self.osu_paths = kept
            self._bundle_cache: OrderedDict[Path, ChartBundle] = OrderedDict()
            self._grid_cache = SharedGridCache(self.grid_dir)
            self._cache_max = min(512, max(64, len(self.osu_paths)))
            self._meta_manifest = (
                load_chart_meta_manifest(manifest_path) if manifest_path.is_file() else None
            )
            if not self.osu_paths:
                raise ValueError("no chart paths for lazy dataset")
        else:
            input_paths = list(osu_paths)
            self.bundles = preload_chart_bundles(
                input_paths,
                grid_dir=self.grid_dir,
                build_grid_if_missing=build_grid_if_missing,
                meta_manifest_path=manifest_path if manifest_path.is_file() else None,
            )
            self.osu_paths = [b.path for b in self.bundles]
            self.lazy = False
            skipped = requested - len(self.bundles)
            if skipped:
                logger.warning(
                    "preload skipped %d / %d charts (cond_vec, grid, parse, or shorter than %d bars)",
                    skipped,
                    requested,
                    WINDOW_BARS,
                )
            if not self.bundles:
                raise ValueError("no chart bundles available for dataset")

    @property
    def n_charts(self) -> int:
        return len(self.osu_paths)

    def __len__(self) -> int:
        return self.n_charts * self.samples_per_chart

    def _bundle_for_index(self, chart_index: int) -> ChartBundle | None:
        chart_index %= self.n_charts
        if not self.lazy:
            return self.bundles[chart_index]
        path = self.osu_paths[chart_index]
        cached = self._bundle_cache.get(path)
        if cached is not None:
            self._bundle_cache.move_to_end(path)
            return cached
        bundle = preload_chart_bundle(
            path,
            grid_dir=self.grid_dir,
            build_grid_if_missing=self._build_grid_if_missing,
            grid_cache=self._grid_cache,
            meta_manifest=self._meta_manifest,
        )
        if bundle is None:
            return None
        if len(self._bundle_cache) >= self._cache_max:
            self._bundle_cache.popitem(last=False)
        self._bundle_cache[path] = bundle
        return bundle

    @staticmethod
    def _sample_to_tensors(sample: TrainingSample) -> dict[str, torch.Tensor]:
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
    ) -> dict[str, torch.Tensor] | None:
        attempts = 1 if self.fixed_start_bar is not None else _MAX_WINDOW_ATTEMPTS
        for attempt in range(attempts):
            rng = random.Random(self.seed + dataset_index + attempt)
            sample = build_sample(
                bundle,
                rng=rng,
                start_bar=self.fixed_start_bar,
            )
            if sample is not None:
                return self._sample_to_tensors(sample)
        return None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        chart_index = index % self.n_charts
        bundle = self._bundle_for_index(chart_index)
        if bundle is None:
            raise ValueError(
                f"could not load chart {self.osu_paths[chart_index]} (dataset index {index})"
            )
        tensors = self._try_sample_bundle(bundle, dataset_index=index)
        if tensors is None:
            raise ValueError(
                f"could not sample a window from {bundle.path} (dataset index {index})"
            )
        return tensors


def pad_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length token windows and audio slices into a batch."""
    max_len = max(item["token_ids"].shape[0] for item in batch)
    max_audio = max(item["audio"].shape[0] for item in batch)
    feat_dim = batch[0]["audio"].shape[1]

    token_ids = torch.full((len(batch), max_len), _PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    cond_vec = torch.stack([item["cond_vec"] for item in batch], dim=0)
    audio = torch.zeros(len(batch), max_audio, feat_dim, dtype=torch.float32)
    audio_mask = torch.zeros(len(batch), max_audio, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["token_ids"].shape[0]
        token_ids[i, :n] = item["token_ids"]
        loss_mask[i, :n] = item["loss_mask"]
        attn_mask[i, :n] = True
        a = item["audio"].shape[0]
        audio[i, :a] = item["audio"]
        audio_mask[i, :a] = True

    return {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "attn_mask": attn_mask,
        "cond_vec": cond_vec,
        "audio": audio,
        "audio_mask": audio_mask,
    }
