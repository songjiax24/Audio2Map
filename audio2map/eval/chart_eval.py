"""Teacher-forcing token metrics and chart-level generation eval."""

from __future__ import annotations

import logging
import math
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from audio2map.dataset.donor import (
    DonorMap,
    assert_exact_donor_universe,
    chart_key,
    default_donor_map_path,
    donor_pool_report,
    require_donor_map,
)
from audio2map.eval.frozen_inputs import DIVERSITY_GENERATION_SEEDS
from audio2map.dataset.sample import TrainingSample, build_sample
from audio2map.eval.cache import cache_get_or_put, generation_cache_key
from audio2map.eval.chart_stats import ChartStatAccumulator
from audio2map.eval.diagnostics import missing_hold_tail_report
from audio2map.eval.diversity_eval import seed_pair_report
from audio2map.eval.motif import MotifAccumulator, MotifKey, parse_motif
from audio2map.eval.motif_freq import MotifCorpus, OrderedMotifVocab, corpus_distance, load_ordered_motif_vocab
from audio2map.eval.note_match import NoteMatchStats, compare_note_lists
from audio2map.eval.result import GeneratedChart
from audio2map.eval.score import score_chart
from audio2map.eval.v0_backend import BACKEND_ID, generate_v0
from audio2map.features.cond import ChartMeta, CondVecError, build_cond_vec, compute_chart_meta
from audio2map.generate.overlap import DecodeConfig, GenerationRangeConfig, OverlapConfig, generate_chart_notes
from audio2map.generate.service import resolve_audio_file
from audio2map.grid import CanonicalTiming
from audio2map.metrics.nll import token_nll_by_type, token_nll_stats
from audio2map.metrics.token import TOKEN_TYPE_BUCKETS, SplitTokenAccuracy, split_token_accuracy
from audio2map.metrics.validity import ValidityStats, logit_validity
from audio2map.model.config import WINDOW_BARS
from audio2map.model.model import AudioChartModel
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.schema import ManiaNote
from audio2map.tokens import invert_vocab
from audio2map.utils.paths import raw_dir

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggregateStats:
    charts: int = 0
    token_correct: int = 0
    token_total: int = 0
    loss_sum: float = 0.0
    split_token: SplitTokenAccuracy = field(default_factory=SplitTokenAccuracy)
    validity: ValidityStats = field(default_factory=ValidityStats)
    type_loss: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in TOKEN_TYPE_BUCKETS}
    )
    type_count: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in TOKEN_TYPE_BUCKETS}
    )

    def to_dict(self) -> dict:
        mean_nll = self.loss_sum / self.token_total if self.token_total else 0.0
        split_nll = {
            name: (self.type_loss[name] / self.type_count[name] if self.type_count[name] else 0.0)
            for name in TOKEN_TYPE_BUCKETS
        }
        return {
            "charts": self.charts,
            "loss": mean_nll,
            "token_acc": self.token_correct / self.token_total if self.token_total else 0.0,
            "split_token_acc": self.split_token.to_dict(),
            "split_nll": split_nll,
            "ppl": math.exp(mean_nll) if self.token_total else None,
            "legal_top1": self.validity.legal_top1,
            "legal_probability": self.validity.legal_probability,
        }


@dataclass(frozen=True, slots=True)
class EvalChart:
    """One chart that produced both a teacher window and a full cond vector."""

    path: Path
    relpath: str
    meta: ChartMeta
    cond_vec: np.ndarray
    sample: TrainingSample


@dataclass(slots=True)
class TeacherEvalResult:
    scenarios: dict[str, AggregateStats]
    charts: list[dict]
    donor_map: str | None = None
    donor_pool: dict | None = None

    def to_dict(self) -> dict:
        return {
            "scenarios": {name: stats.to_dict() for name, stats in self.scenarios.items()},
            "charts": self.charts,
            "donor_map": self.donor_map,
            "donor_pool": self.donor_pool,
        }


def materialize_eval_charts(
    paths: list[Path],
    *,
    rng: random.Random,
    build_grid_if_missing: bool,
    fixed_start_bar: int | None,
    grid_dir: Path | None,
    raw_root: Path,
    need_relpath: bool,
) -> list[EvalChart]:
    """Shared teacher/generation pool.

    A chart stays in the pool only when ``compute_chart_meta`` yields a full
    ``cond_vec`` and ``build_sample`` succeeds. Both eval modes use this list,
    so the donor candidate set does not depend on which mode creates the map.
    """
    prepared: list[EvalChart] = []
    for path in paths:
        meta = compute_chart_meta(path)
        try:
            build_cond_vec(meta)
        except CondVecError as exc:
            logger.warning("%s", exc)
            continue
        sample = build_sample(
            path,
            rng=rng,
            build_grid_if_missing=build_grid_if_missing,
            start_bar=fixed_start_bar,
            grid_dir=grid_dir,
            chart_meta=meta,
        )
        if sample is None:
            logger.warning("skip eval chart, cannot build window: %s", path)
            continue
        relpath = chart_key(path, raw_root) if need_relpath else ""
        prepared.append(
            EvalChart(
                path=path,
                relpath=relpath,
                meta=meta,
                cond_vec=sample.cond_vec,
                sample=sample,
            )
        )
    return prepared


def assert_materialization_complete(requested: list[Path], prepared: list[EvalChart]) -> None:
    """Fail when a requested chart did not survive the shared materialization gate.

    Teacher and generation both use that gate. A partial pool is not evaluated.
    """
    prepared_paths = {chart.path.resolve() for chart in prepared}
    missing = [str(path) for path in requested if Path(path).resolve() not in prepared_paths]
    if missing:
        raise ValueError(
            f"{len(missing)} chart(s) failed compute_chart_meta, build_cond_vec, or build_sample "
            f"and were not evaluated: {missing[:20]}"
        )


def pin_official_candidate_pool(relpaths: list[str], donor_map_path: Path, *, seed: int) -> DonorMap:
    """Load the frozen donor map and require it to be exactly this pool."""
    mapping = require_donor_map(donor_map_path, relpaths, seed=seed)
    assert_exact_donor_universe(mapping, relpaths)
    return mapping


def _reject_unconstrained_scope(scenarios: tuple[str, ...], modes: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    if "unconstrained" not in modes:
        return
    if any(scenario != "matched_full" for scenario in scenarios):
        raise ValueError("unconstrained validity runs only for matched_full")
    if tuple(int(seed) for seed in seeds) != (0,):
        raise ValueError("unconstrained validity runs only for generation seed 0")


def _score_teacher_sample(
    model: AudioChartModel,
    sample: TrainingSample,
    cond_vec: np.ndarray,
    *,
    device: torch.device,
    agg: AggregateStats,
    id_to_token: dict[int, str],
    window_bars: int,
) -> None:
    agg.charts += 1
    token_ids = torch.tensor([sample.token_ids], dtype=torch.long, device=device)
    loss_mask = torch.tensor([sample.loss_mask], dtype=torch.float32, device=device)
    audio = torch.tensor([sample.audio_features], dtype=torch.float32, device=device)
    cond = torch.tensor([cond_vec], dtype=torch.float32, device=device)
    attn = torch.ones(1, token_ids.shape[1], dtype=torch.bool, device=device)
    logits = model(audio, cond, token_ids, attn_mask=attn)
    targets = token_ids[:, 1:]
    mask = loss_mask[:, 1:]
    loss_sum, valid, correct = token_nll_stats(logits, targets, mask)
    agg.loss_sum += float(loss_sum.item())
    agg.token_correct += int(correct.item())
    agg.token_total += int(valid.item())
    for name, (bucket_loss, bucket_count) in token_nll_by_type(
        logits, targets, mask, id_to_token=id_to_token
    ).items():
        agg.type_loss[name] += bucket_loss
        agg.type_count[name] += bucket_count
    pred = logits.argmax(dim=-1)
    agg.split_token.merge(
        split_token_accuracy(pred[0], targets[0], mask[0], id_to_token=id_to_token)
    )
    agg.validity.merge(
        logit_validity(
            logits,
            token_ids,
            loss_mask,
            window_bars=window_bars,
            id_to_token=id_to_token,
        )
    )


def eval_teacher_forcing(
    model: AudioChartModel,
    paths: list[Path],
    *,
    device: torch.device,
    seed: int = 0,
    build_grid_if_missing: bool = False,
    fixed_start_bar: int | None = None,
    grid_dir: Path | None = None,
    window_bars: int = WINDOW_BARS,
    scenarios: tuple[str, ...] = ("matched_full",),
    donor_map_path: Path | None = None,
    donor_seed: int | None = None,
    raw_root: Path | None = None,
    official_exact: bool = False,
) -> TeacherEvalResult:
    """Token accuracy with GT prefix. Windows are built once per chart.

    ``random_full`` replaces only the condition vector, using the persistent
    target → donor path map. The window RNG is separate from the donor RNG.
    """
    unknown = [name for name in scenarios if name not in ("matched_full", "random_full")]
    if unknown:
        raise ValueError(f"unsupported teacher scenario: {unknown}")
    root = raw_root or raw_dir()
    rng = random.Random(seed)
    prepared = materialize_eval_charts(
        paths,
        rng=rng,
        build_grid_if_missing=build_grid_if_missing,
        fixed_start_bar=fixed_start_bar,
        grid_dir=grid_dir,
        raw_root=root,
        need_relpath="random_full" in scenarios or official_exact,
    )
    assert_materialization_complete(paths, prepared)
    by_rel = {chart.relpath: chart for chart in prepared}
    donor: DonorMap | None = None
    donor_path: Path | None = None
    if official_exact:
        if not prepared:
            raise ValueError("no charts materialized for the official candidate pool")
        if donor_seed is None:
            raise ValueError("official candidate-pool check requires the frozen donor-map seed")
        donor_path = donor_map_path or default_donor_map_path()
        pinned = pin_official_candidate_pool(
            [chart.relpath for chart in prepared],
            donor_path,
            seed=donor_seed,
        )
        if "random_full" in scenarios:
            donor = pinned
    if "random_full" in scenarios and donor is None:
        if not prepared:
            raise ValueError("no charts materialized for donor mapping")
        donor_path = donor_map_path or default_donor_map_path()
        donor = require_donor_map(
            donor_path,
            [chart.relpath for chart in prepared],
            seed=donor_seed,
        )
    if donor is not None:
        missing_donors = sorted(
            donor.pairs[chart.relpath] for chart in prepared if donor.pairs[chart.relpath] not in by_rel
        )
        if missing_donors:
            raise ValueError(
                "donor chart is not in this eval's materialized pool "
                f"(refusing to resample): {missing_donors[:20]}"
            )

    id_to_token = invert_vocab()
    model = model.to(device)
    model.eval()
    scenarios_out: dict[str, AggregateStats] = {}
    with torch.no_grad():
        for scenario in scenarios:
            agg = AggregateStats()
            for chart in prepared:
                cond_vec = chart.sample.cond_vec
                if scenario == "random_full":
                    assert donor is not None
                    cond_vec = by_rel[donor.pairs[chart.relpath]].sample.cond_vec
                _score_teacher_sample(
                    model,
                    chart.sample,
                    cond_vec,
                    device=device,
                    agg=agg,
                    id_to_token=id_to_token,
                    window_bars=window_bars,
                )
            scenarios_out[scenario] = agg

    chart_rows = []
    for chart in prepared:
        row = {"osu": str(chart.path), "relpath": chart.relpath}
        if donor is not None:
            row["donor"] = donor.pairs[chart.relpath]
        chart_rows.append(row)
    pool = None
    if donor is not None:
        targets = [chart.relpath for chart in prepared]
        pool = donor_pool_report(donor, targets)
        logger.info(
            "donor pool candidates=%d universe=%s",
            pool["candidate_count"],
            pool["universe_id"],
        )
        for row in pool["pairs"]:
            logger.info("target %s donor %s", row["target"], row["donor"])
    return TeacherEvalResult(
        scenarios=scenarios_out,
        charts=chart_rows,
        donor_map=str(donor_path) if donor_path is not None else None,
        donor_pool=pool,
    )


def eval_inference_chart(
    model: AudioChartModel,
    path: Path,
    *,
    device: torch.device,
    cond_vec: np.ndarray,
    overlap: OverlapConfig | None = None,
    decode: DecodeConfig | None = None,
    build_grid_if_missing: bool = True,
    range_mode: Literal["audio_full", "reference_chart"] = "audio_full",
    post_margin_bars: int = 4,
    grid_dir: Path | None = None,
) -> NoteMatchStats:
    beatmap = parse_beatmap(path)
    timing = CanonicalTiming.from_beatmap(beatmap)
    pred, _ = generate_chart_notes(
        model,
        audio_path=resolve_audio_file(path),
        timing=timing,
        cond_vec=cond_vec,
        grid_dir=grid_dir,
        overlap=overlap or OverlapConfig(),
        range_cfg=GenerationRangeConfig(mode=range_mode, post_margin_bars=post_margin_bars),
        device=device,
        decode=decode or DecodeConfig(),
        reference_notes=beatmap.notes,
        build_grid_if_missing=build_grid_if_missing,
    )
    return compare_note_lists(pred, beatmap.notes, timing)


def _motif_vocab_from_json(data: dict | None) -> dict[str, set[MotifKey]] | None:
    if not data:
        return None
    return {scope: {parse_motif(text) for text in items} for scope, items in data.items()}


def _ordered_motif_vocab(data: dict | None) -> OrderedMotifVocab | None:
    if not data:
        return None
    return load_ordered_motif_vocab(data)


def eval_generation(
    model: AudioChartModel,
    charts: list[tuple[Path, ChartMeta, np.ndarray]],
    *,
    device: torch.device,
    scenarios: tuple[str, ...] = ("matched_full",),
    modes: tuple[str, ...] = ("constrained",),
    seeds: tuple[int, ...] = (0,),
    donor: DonorMap | None = None,
    raw_root: Path | None = None,
    diversity_rels: list[str] | None = None,
    unconstrained_rels: list[str] | None = None,
    motif_vocab: dict | None = None,
    decode: DecodeConfig | None = None,
    build_grid_if_missing: bool = True,
    range_mode: Literal["audio_full", "reference_chart"] = "audio_full",
    post_margin_bars: int = 4,
    grid_dir: Path | None = None,
    adherence_dir: Path | None = None,
) -> dict:
    """Constrained notes feed match, adherence, statistics, motif, and diversity.

    Unconstrained runs only produce chart validity, and only for matched_full
    at generation seed 0. The two are not mixed. Statistics and motifs are
    accumulated per seed on ``seeds``. Diversity, when ``diversity_rels`` is
    set, compares the fixed seeds ``0,1,2,3,4`` on that pinned list. Those
    seeds do not follow ``seeds``. Seed 0 reuses the matched constrained
    generation from this call when seed 0 is also in ``seeds``.

    This function is the v0 orchestration. v1 scores a ``GeneratedChart``
    through ``score_chart`` and does not call it.
    """
    _reject_unconstrained_scope(scenarios, modes, seeds)
    root = raw_root or raw_dir()
    by_rel = {chart_key(path, root): (path, meta, cond) for path, meta, cond in charts}
    if "random_full" in scenarios:
        if donor is None:
            raise ValueError("random_full generation requires a donor map")
        absent = sorted(rel for rel in by_rel if rel not in donor.pairs)
        if absent:
            raise ValueError(f"target chart is missing from the donor map: {absent[:20]}")
        outside = sorted(
            rel for rel in by_rel if donor.pairs[rel] not in by_rel
        )
        if outside:
            raise ValueError(
                "donor chart is not in this eval's materialized pool "
                f"(refusing to resample): {outside[:20]}"
            )
    vocab = _motif_vocab_from_json(motif_vocab)
    ordered_vocab = _ordered_motif_vocab(motif_vocab)
    decode = decode or DecodeConfig()
    overlap = OverlapConfig()
    pre_margin_bars = GenerationRangeConfig().pre_margin_bars
    if "unconstrained" in modes:
        if not unconstrained_rels:
            raise ValueError("unconstrained generation requires a pinned chart subset")
        absent_subset = [rel for rel in unconstrained_rels if rel not in by_rel]
        if absent_subset:
            raise ValueError(
                f"unconstrained subset is not in this eval pool: {absent_subset[:20]}"
            )
    cache: dict[tuple, tuple] = {}
    owned_tmp = None
    if adherence_dir is None:
        owned_tmp = tempfile.TemporaryDirectory()
        adherence_dir = Path(owned_tmp.name)
    adherence_root = adherence_dir

    def run(path: Path, cond: np.ndarray, mode: str, seed: int, scenario: str):
        rel = chart_key(path, root)
        key = generation_cache_key(
            backend=BACKEND_ID,
            relpath=rel,
            scenario=scenario,
            mode=mode,
            seed=seed,
            cond_vec=cond,
            temperature=decode.temperature,
            top_p=decode.top_p,
            top_k=decode.top_k,
            range_mode=range_mode,
            pre_margin_bars=pre_margin_bars,
            post_margin_bars=post_margin_bars,
            window_bars=overlap.window_bars,
            context_bars=overlap.context_bars,
            keep_bars=overlap.keep_bars,
            future_bars=overlap.future_bars,
            max_seq_len=overlap.max_seq_len,
        )

        def produce():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            beatmap = parse_beatmap(path)
            timing = CanonicalTiming.from_beatmap(beatmap)
            notes, validity, issues = generate_v0(
                model,
                path=path,
                timing=timing,
                cond_vec=cond,
                device=device,
                decode=decode,
                constrain=mode == "constrained",
                grid_dir=grid_dir,
                build_grid_if_missing=build_grid_if_missing,
                range_mode=range_mode,
                post_margin_bars=post_margin_bars,
                reference_notes=beatmap.notes,
                overlap=overlap,
                pre_margin_bars=pre_margin_bars,
            )
            return notes, validity, timing, issues

        return cache_get_or_put(cache, key, produce)

    out: dict = {"scenarios": {}, "diversity": {}}
    try:
        out = _generation_body(
            run,
            charts,
            scenarios=scenarios,
            modes=modes,
            seeds=seeds,
            donor=donor,
            root=root,
            by_rel=by_rel,
            diversity_rels=diversity_rels,
            unconstrained_rels=unconstrained_rels,
            vocab=vocab,
            ordered_vocab=ordered_vocab,
            adherence_root=adherence_root,
        )
    finally:
        if owned_tmp is not None:
            owned_tmp.cleanup()
    if donor is not None:
        out["donor_pool"] = donor_pool_report(donor, list(by_rel))
        logger.info(
            "donor pool candidates=%d universe=%s",
            out["donor_pool"]["candidate_count"],
            out["donor_pool"]["universe_id"],
        )
        for row in out["donor_pool"]["pairs"]:
            logger.info("target %s donor %s", row["target"], row["donor"])
    return out


def _condition_for(path: Path, meta: ChartMeta, cond: np.ndarray, scenario: str, donor, by_rel, root):
    donor_rel = None
    adherence_meta = meta
    used = cond
    if scenario == "random_full":
        assert donor is not None
        rel = chart_key(path, root)
        donor_rel = donor.pairs[rel]
        adherence_meta = by_rel[donor_rel][1]
        used = by_rel[donor_rel][2]
    return used, donor_rel, adherence_meta


def _generation_body(
    run,
    charts,
    *,
    scenarios,
    modes,
    seeds,
    donor,
    root,
    by_rel,
    diversity_rels,
    unconstrained_rels,
    vocab,
    ordered_vocab: OrderedMotifVocab | None,
    adherence_root: Path,
) -> dict:
    _reject_unconstrained_scope(scenarios, modes, seeds)
    if diversity_rels:
        missing = [rel for rel in diversity_rels if rel not in by_rel]
        if missing:
            raise ValueError(f"diversity subset is not in this eval pool: {missing[:20]}")
    out: dict = {"scenarios": {}, "diversity": {}}
    unconstrained_order = list(unconstrained_rels or [])
    for scenario in scenarios:
        scenario_body: dict = {}
        for mode in modes:
            if mode not in ("constrained", "unconstrained"):
                raise ValueError(f"unknown generation mode: {mode}")
            if mode == "unconstrained":
                per_seed = []
                for seed in seeds:
                    labels = []
                    for rel in unconstrained_order:
                        path, _meta, cond = by_rel[rel]
                        used, donor_rel, _adherence = _condition_for(
                            path, _meta, cond, scenario, donor, by_rel, root
                        )
                        _notes, validity, _timing, _issues = run(path, used, mode, seed, scenario)
                        labels.append(
                            {
                                "osu": str(path),
                                "donor": donor_rel,
                                "validity": validity,
                                "decode_issues": _issues,
                            }
                        )
                    n = len(labels)
                    counts = {
                        name: sum(row["validity"] == name for row in labels)
                        for name in ("valid", "illegal_token", "truncated")
                    }
                    per_seed.append(
                        {
                            "seed": seed,
                            "charts": labels,
                            "valid_rate": counts["valid"] / n if n else 0.0,
                            "illegal_token_rate": counts["illegal_token"] / n if n else 0.0,
                            "truncated_rate": counts["truncated"] / n if n else 0.0,
                        }
                    )
                scenario_body["unconstrained"] = per_seed
                continue

            human_motifs = None
            if ordered_vocab is not None:
                human_motifs = MotifCorpus(ordered_vocab)
                for path, _meta, _cond in charts:
                    beatmap = parse_beatmap(path)
                    human_motifs.add_chart(
                        beatmap.notes, CanonicalTiming.from_beatmap(beatmap)
                    )
            per_seed_constrained = []
            for seed in seeds:
                stats = ChartStatAccumulator()
                motifs = MotifAccumulator(vocab)
                generated_motifs = MotifCorpus(ordered_vocab) if ordered_vocab is not None else None
                chart_rows = []
                issue_rows: list[list[dict]] = []
                for path, meta, cond in charts:
                    used, donor_rel, adherence_meta = _condition_for(
                        path, meta, cond, scenario, donor, by_rel, root
                    )
                    notes, validity, timing, issues = run(path, used, mode, seed, scenario)
                    reference = parse_beatmap(path).notes
                    generated = GeneratedChart(
                        relpath=chart_key(path, root),
                        path=path,
                        notes=notes,
                        timing=timing,
                        reference_notes=reference,
                        target_meta=meta,
                        adherence_meta=adherence_meta,
                        validity=validity,
                        decode_issues=issues,
                        scenario=scenario,
                        mode=mode,
                        seed=seed,
                        donor=donor_rel,
                    )
                    scored = score_chart(
                        generated,
                        adherence_root / f"{len(chart_rows)}-{scenario}-{seed}.osu",
                    )
                    if scored is None:
                        continue
                    stats.add_chart(notes, timing)
                    motifs.add_chart(notes, timing)
                    if generated_motifs is not None:
                        generated_motifs.add_chart(notes, timing)
                    issue_rows.append(issues)
                    chart_rows.append(scored)
                seed_row = {
                    "seed": seed,
                    "charts": chart_rows,
                    "statistics": stats.to_dict(),
                    # ``motifs`` is the per-lane average-count dict, sorted by motif
                    # string. ``motif_distribution`` and ``motif_distance`` follow
                    # the frozen list order and are the official counting result.
                    "motifs": motifs.to_dict(),
                    "missing_hold_tail": missing_hold_tail_report(issue_rows),
                }
                if generated_motifs is not None and human_motifs is not None:
                    seed_row["motif_distribution"] = generated_motifs.as_dict()
                    seed_row["motif_distance"] = corpus_distance(generated_motifs, human_motifs)
                per_seed_constrained.append(seed_row)
            scenario_body["constrained"] = per_seed_constrained
        out["scenarios"][scenario] = scenario_body

        if scenario == "matched_full" and diversity_rels and "constrained" in modes:
            div_rows = []
            for rel in diversity_rels:
                path, _meta, cond = by_rel[rel]
                generated = []
                frequencies = []
                observations = []
                issue_by_seed = []
                timing = None
                for seed in DIVERSITY_GENERATION_SEEDS:
                    notes, _validity, timing, issues = run(
                        path, cond, "constrained", seed, scenario
                    )
                    generated.append(notes)
                    issue_by_seed.append(issues)
                    if ordered_vocab is not None and timing is not None:
                        one = MotifCorpus(ordered_vocab)
                        one.add_chart(notes, timing)
                        frequencies.append(
                            {scope: one.frequency(scope) for scope in ordered_vocab.texts}
                        )
                        observations.append(dict(one.observations))
                assert timing is not None
                div_rows.append(
                    {
                        "osu": str(path),
                        "decode_issues": issue_by_seed,
                        **seed_pair_report(
                            generated,
                            timing,
                            frequencies=frequencies or None,
                            observations=observations or None,
                            vocab=ordered_vocab,
                        ),
                    }
                )
            out["diversity"][scenario] = div_rows
    return out
