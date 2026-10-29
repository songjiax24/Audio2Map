"""Build one chart on a worker. Augment and the condition mask are already drawn."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

import torch

from audio2map.dataset.v1.augment import AugmentChoice, clock_chart
from audio2map.dataset.v1.example import ChartAudio, build_v1_example, load_chart_audio
from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.store import rate1_meta_for
from audio2map.model.v1.batch import V1Batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlannedChart:
    """One chart after the main process has consumed the train RNG."""

    relpath: str
    mirror: bool
    rate: float
    meta: dict | None
    cond_mask: list[float] | None


@dataclass
class InFlight:
    """A batch whose RNG draws are done and whose builds may still be running."""

    end_pos: int
    rng_after: dict
    plans: list[PlannedChart | None]
    futures: list[Future[V1Batch] | None]


def build_planned(root: str, planned: PlannedChart) -> V1Batch:
    """Decode, clock, and featurize. Does not draw augment or a condition mask."""
    from pathlib import Path

    loaded = load_chart_audio(Path(root) / planned.relpath)
    t_audio = loaded.t_audio_ms
    if t_audio is None:
        t_audio = float(loaded.pcm.shape[0]) / float(SAMPLE_RATE) * 1000.0
    prepared = clock_chart(
        list(loaded.notes),
        loaded.redlines,
        loaded.pcm,
        t_audio,
        AugmentChoice(mirror=planned.mirror, rate=planned.rate),
    )
    source = ChartAudio(notes=prepared.notes, redlines=prepared.redlines, pcm=prepared.pcm)
    cond: torch.Tensor | str
    if planned.cond_mask is None:
        cond = "none"
    else:
        cond = torch.tensor(planned.cond_mask, dtype=torch.float32)
    if planned.meta is None:
        if planned.rate != 1.0:
            raise RuntimeError("rate-1 condition is required before a rate change")
        meta = rate1_meta_for(Path(root), planned.relpath)
    else:
        meta = ChartMeta.from_dict(planned.meta)
    return build_v1_example(
        source,
        meta=meta,
        cond_mask=cond,
        encoded=prepared.encoded,
    )


class ChartBuilder:
    """Draw plans on this process. Decode in worker processes when ``workers > 1``."""

    def __init__(self, dataset, workers: int) -> None:
        if workers < 1:
            raise ValueError(f"workers must be positive, got {workers}")
        self.dataset = dataset
        self.root = str(dataset.root)
        self.pool: ProcessPoolExecutor | None = None
        if workers > 1:
            context = mp.get_context("spawn")
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=context)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

    def submit(self, indices: list[int], end_pos: int) -> InFlight:
        plans: list[PlannedChart | None] = []
        for index in indices:
            relpath = self.dataset.charts[index].relpath
            try:
                plans.append(self.dataset.plan(index))
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("skip %s: %s", relpath, exc)
                plans.append(None)
        rng_after = copy.deepcopy(self.dataset.rng_state())
        futures: list[Future[V1Batch] | None] = []
        for plan in plans:
            if plan is None or self.pool is None:
                futures.append(None)
                continue
            futures.append(self.pool.submit(build_planned, self.root, plan))
        return InFlight(end_pos=end_pos, rng_after=rng_after, plans=plans, futures=futures)

    def wait(self, flight: InFlight) -> list[V1Batch | None]:
        built: list[V1Batch | None] = []
        for plan, future in zip(flight.plans, flight.futures, strict=True):
            if plan is None:
                built.append(None)
                continue
            try:
                if future is None:
                    built.append(build_planned(self.root, plan))
                else:
                    built.append(future.result())
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning("skip %s: %s", plan.relpath, exc)
                built.append(None)
        return built
