"""One whole chart per index. Augmentation is on only for the training split."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audio2map.dataset.split import ChartSnapshot
from audio2map.dataset.v1.augment import clock_chart, sample_augment
from audio2map.dataset.v1.example import ChartAudio, CondMask, build_v1_example, load_chart_audio
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta, compute_chart_meta
from audio2map.features.cond.v1.mask import sample_cond_mask
from audio2map.features.cond.v1.rate import scalars_at_rate
from audio2map.features.cond.v1.store import append_v1_chart_meta_records, load_v1_chart_meta
from audio2map.features.cond.v1.vec import build_cond
from audio2map.model.v1.batch import V1Batch
from audio2map.train.v1.prepare import PlannedChart

logger = logging.getLogger(__name__)


class V1ChartDataset(Dataset):
    """Build a ``V1Batch`` of size 1. Rate-1 metadata is cached on this worker."""

    def __init__(
        self,
        charts: list[ChartSnapshot],
        *,
        root: Path,
        augment: bool,
        cond_mask: CondMask,
        seed: int,
    ) -> None:
        self.charts = list(charts)
        self.root = root
        self.augment = augment
        self.cond_mask = cond_mask
        self._rng = np.random.default_rng(seed)
        self._meta: dict[str, ChartMeta] = {}
        self._meta_loaded = False

    def __len__(self) -> int:
        return len(self.charts)

    def rng_state(self) -> dict:
        """Numpy bit-generator state for augment and condition-mask draws."""
        return self._rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self._rng.bit_generator.state = state

    def plan(self, index: int) -> PlannedChart:
        """Draw augment and the condition mask. The worker does not touch ``_rng``."""
        chart = self.charts[index]
        path = self.root / chart.relpath
        choice = sample_augment(self._rng, enabled=self.augment)
        # Validation draws neither augment nor a mask, so the worker computes
        # the rate-1 condition and overlaps it with decode. Training needs the
        # condition before the mask draw, on this process.
        meta_now = self.cond_mask == "train" or choice.rate != 1.0
        meta = self._meta_at(chart.relpath, path, choice.rate) if meta_now else None
        cond_mask: list[float] | None
        if self.cond_mask == "train":
            if meta is None:
                raise RuntimeError("training condition is missing before the mask draw")
            drawn = sample_cond_mask(build_cond(meta).m_real, self._rng)
            cond_mask = [float(value) for value in drawn.tolist()]
        elif isinstance(self.cond_mask, torch.Tensor):
            cond_mask = [float(value) for value in self.cond_mask.tolist()]
        else:
            cond_mask = None
        return PlannedChart(
            relpath=chart.relpath,
            mirror=choice.mirror,
            rate=choice.rate,
            meta=None if meta is None else meta.to_dict(),
            cond_mask=cond_mask,
        )

    def __getitem__(self, index: int) -> V1Batch:
        chart = self.charts[index]
        path = self.root / chart.relpath
        choice = sample_augment(self._rng, enabled=self.augment)
        meta = self._meta_at(chart.relpath, path, choice.rate)
        loaded = load_chart_audio(path)
        t_audio = loaded.t_audio_ms
        if t_audio is None:
            t_audio = float(loaded.pcm.shape[0]) / float(SAMPLE_RATE) * 1000.0
        prepared = clock_chart(
            list(loaded.notes),
            loaded.redlines,
            loaded.pcm,
            t_audio,
            choice,
        )
        source = ChartAudio(
            notes=prepared.notes,
            redlines=prepared.redlines,
            pcm=prepared.pcm,
        )
        return build_v1_example(
            source,
            meta=meta,
            cond_mask=self.cond_mask,
            rng=self._rng,
            encoded=prepared.encoded,
        )

    def _ensure_disk(self) -> None:
        if self._meta_loaded:
            return
        self._meta.update(load_v1_chart_meta())
        self._meta_loaded = True

    def _meta_at(self, relpath: str, path: Path, rate: float) -> ChartMeta:
        self._ensure_disk()
        base = self._meta.get(relpath)
        if base is None:
            base = compute_chart_meta(path)
            self._meta[relpath] = base
            record = base.to_dict()
            record["relpath"] = relpath
            append_v1_chart_meta_records([record])
            if base.error:
                logger.warning("chart meta %s: %s", path.name, base.error)
        if rate == 1.0:
            return base
        return scalars_at_rate(path, base, rate)
