"""``audio2map-eval``: teacher-forcing / inference eval."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import torch

from audio2map.cli.common import parse_args_with_config, setup_logging
from audio2map.dataset.bundle import filter_v0_paths
from audio2map.dataset.chart_subset import (
    default_diversity_subset_path,
    default_unconstrained_subset_path,
    require_chart_subset,
)
from audio2map.dataset.donor import (
    chart_key,
    default_donor_map_path,
    require_donor_map,
)
from audio2map.eval.frozen_inputs import (
    DIVERSITY_SUBSET_COUNT,
    DIVERSITY_SUBSET_SEED,
    RANDOM_FULL_SEED,
    UNCONSTRAINED_SUBSET_COUNT,
    UNCONSTRAINED_SUBSET_SEED,
    assert_motif_list_count,
    default_motif_lists_path,
)
from audio2map.dataset.split import (
    GROUP_BY,
    SPLIT_SCHEME,
    SplitError,
    SplitName,
    default_split_manifest_path,
    group_id_for_chart,
    list_raw_osu_paths,
    load_split_manifest,
    split_group_chart_counts,
    split_paths,
)
from audio2map.eval.chart_eval import (
    assert_materialization_complete,
    eval_generation,
    eval_inference_chart,
    eval_teacher_forcing,
    materialize_eval_charts,
    pin_official_candidate_pool,
)
from audio2map.features.cond import build_cond_vec, compute_chart_meta
from audio2map.generate.overlap import DecodeConfig, GenerationRangeConfig
from audio2map.model.config import WINDOW_BARS
from audio2map.model.model import load_checkpoint
from audio2map.utils.paths import audio_grid_dir, raw_dir

logger = logging.getLogger(__name__)

_DEFAULT_DECODE = DecodeConfig()


def _teacher_paths(args: argparse.Namespace) -> list[Path]:
    if args.osu:
        path = Path(args.osu)
        if not path.is_file():
            raise SystemExit(f"osu file not found: {path}")
        if args.split is None:
            return [path]
        try:
            manifest, raw_root = _load_split(args)
            group_id = group_id_for_chart(path, raw_root)
            assigned = manifest.groups.get(group_id)
        except SplitError as exc:
            raise SystemExit(str(exc)) from exc
        if assigned is None or assigned != args.split:
            raise SystemExit(
                f"{path} group {group_id} is split {assigned!r}, expected {args.split!r}"
            )
        return [path]

    split_name: SplitName = args.split or "test"
    try:
        manifest, raw_root = _load_split(args)
        raw_paths = list_raw_osu_paths(raw_root)
        by_split = split_paths(raw_paths, manifest, raw_root=raw_root)
    except SplitError as exc:
        raise SystemExit(str(exc)) from exc
    counts = split_group_chart_counts(manifest)
    n_g, n_c = counts[split_name]
    logger.info(
        "split groups/charts train=%d/%d val=%d/%d test=%d/%d; using %s",
        counts["train"][0],
        counts["train"][1],
        counts["val"][0],
        counts["val"][1],
        counts["test"][0],
        counts["test"][1],
        split_name,
    )
    grid_dir = Path(args.grid_dir) if args.grid_dir else audio_grid_dir()
    require_grid = not args.build_grid
    paths = filter_v0_paths(
        by_split[split_name],
        require_grid=require_grid,
        grid_dir=grid_dir,
        min_window_bars=WINDOW_BARS,
    )
    logger.info(
        "v0-eligible %s charts=%d (of %d in split, %d groups, require_grid=%s)",
        split_name,
        len(paths),
        n_c,
        n_g,
        require_grid,
    )
    paths = sorted(paths, key=lambda path: path.as_posix())
    if args.limit is not None and args.limit < len(paths):
        rng = random.Random(args.seed)
        paths = sorted(rng.sample(paths, args.limit), key=lambda path: path.as_posix())
        logger.info("subsampled to %d charts (--limit)", len(paths))
    return paths


def _official_full_test(args: argparse.Namespace) -> bool:
    """Full v0 test split, with no single-chart or subsample override."""
    if args.osu is not None or args.limit is not None:
        return False
    return (args.split or "test") == "test"


def _reject_custom_frozen_inputs(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    flagged = [name for name in names if getattr(args, name)]
    if flagged:
        flags = ", ".join("--" + name.replace("_", "-") for name in flagged)
        raise SystemExit(f"official test eval reads the frozen inputs only; remove {flags}")


def _load_split(args: argparse.Namespace):
    split_path = (
        Path(args.split_manifest) if args.split_manifest else default_split_manifest_path()
    )
    manifest = load_split_manifest(
        split_path,
        expected_scheme=SPLIT_SCHEME,
        expected_group_by=GROUP_BY,
    )
    return manifest, raw_dir()


def main() -> None:
    p = argparse.ArgumentParser(description="Audio2Map evaluation")
    p.add_argument("--config", type=str, default=None, help="YAML config (CLI flags win)")
    p.add_argument(
        "--mode",
        choices=("teacher", "infer", "generate"),
        required=True,
        help="teacher=token NLL; infer=one-chart debug; generate=constrained metrics and unconstrained validity",
    )
    p.add_argument(
        "--scenario",
        action="append",
        choices=("matched_full", "random_full"),
        default=None,
        help="condition scenario (repeatable; default: matched_full)",
    )
    p.add_argument(
        "--donor-map",
        type=str,
        default=None,
        help=(
            "frozen target→donor JSON, seed 2028 "
            "(default: DATA/splits/random_full.json). "
            "A full test eval also uses it to pin the candidate pool. The file is not created"
        ),
    )
    p.add_argument(
        "--generation-mode",
        action="append",
        choices=("constrained", "unconstrained"),
        default=None,
        help="generate: decoding mode (repeatable; default: constrained)",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0",
        help=(
            "generate: comma-separated seeds for full-pool statistics. "
            "When diversity runs, it compares seeds 0,1,2,3,4 and does not follow this list. "
            "Unconstrained accepts only seed 0"
        ),
    )
    p.add_argument(
        "--diversity-charts",
        type=int,
        default=None,
        help=(
            "generate: count check for the frozen diversity subset. "
            "A full test generate loads that 100-chart file unless this is 0. "
            "The file is not redrawn"
        ),
    )
    p.add_argument(
        "--diversity-seed",
        type=int,
        default=0,
        help="ignored: subset membership is frozen at seed 2026; diversity generation uses seeds 0,1,2,3,4",
    )
    p.add_argument(
        "--diversity-osu",
        action="append",
        default=None,
        help="generate: explicit diversity charts. They must match the frozen subset. The file is not written",
    )
    p.add_argument(
        "--diversity-subset",
        type=str,
        default=None,
        help="generate: pinned diversity JSON (default: DATA/splits/diversity_subset.json)",
    )
    p.add_argument(
        "--motif-vocab",
        type=str,
        default=None,
        help=(
            "generate: ordered lane/hand/row lists of 50000 motifs "
            "(default: freeze/motif_topk_fit_50000.lists.json). Shorter lists are rejected"
        ),
    )
    p.add_argument(
        "--unconstrained-subset",
        type=str,
        default=None,
        help="generate: frozen 300-chart unconstrained list, seed 2027 (default: DATA/splits/unconstrained_subset.json)",
    )
    p.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="split JSON (default: DATA/splits/beatmapset_v1.json)",
    )
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="teacher: frozen split to evaluate (default: test). With --osu, require that chart's group.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="teacher: optional subsample of the selected split (default: all)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--build-grid", action="store_true", help="build audio grid on the fly (slow)")
    p.add_argument("--osu", type=str, default=None, help="single chart (required for infer)")
    p.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="audio_grid cache dir (must match training; default: processed/audio_grid)",
    )
    p.add_argument("--start-bar", type=int, default=None, help="fixed window start (teacher)")
    p.add_argument(
        "--range-mode",
        choices=("audio_full", "reference_chart"),
        default="audio_full",
        help="infer: audio_full=whole audio; reference_chart=clip to original chart bars",
    )
    p.add_argument("--post-margin-bars", type=int, default=GenerationRangeConfig().post_margin_bars)
    p.add_argument("--temperature", type=float, default=_DEFAULT_DECODE.temperature)
    p.add_argument("--top-p", type=float, default=_DEFAULT_DECODE.top_p)
    p.add_argument("--top-k", type=int, default=_DEFAULT_DECODE.top_k)
    p.add_argument("--greedy", action="store_true", help="infer: temperature=0")
    args = parse_args_with_config(p)

    setup_logging()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = load_checkpoint(Path(args.checkpoint), device)

    scenarios = tuple(args.scenario or ("matched_full",))
    donor_map = Path(args.donor_map) if args.donor_map else None
    grid_dir = Path(args.grid_dir) if args.grid_dir else None
    official = _official_full_test(args)
    if official and args.mode == "generate":
        _reject_custom_frozen_inputs(
            args,
            ("donor_map", "diversity_subset", "unconstrained_subset", "motif_vocab"),
        )
    elif official and args.mode == "teacher":
        _reject_custom_frozen_inputs(args, ("donor_map",))

    if args.mode == "infer":
        if not args.osu:
            raise SystemExit("infer mode requires --osu")
        path = Path(args.osu)
        meta = compute_chart_meta(path)
        cond = build_cond_vec(meta)
        donor_rel = None
        if "random_full" in scenarios:
            from audio2map.utils.paths import raw_dir

            root = raw_dir()
            map_path = donor_map or default_donor_map_path()
            rel = chart_key(path, root)
            try:
                mapping = require_donor_map(
                    map_path,
                    [rel],
                    seed=RANDOM_FULL_SEED,
                )
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            donor_rel = mapping.pairs[rel]
            cond = build_cond_vec(compute_chart_meta(root / donor_rel))
        stats = eval_inference_chart(
            model,
            path,
            device=device,
            cond_vec=cond,
            build_grid_if_missing=args.build_grid,
            range_mode=args.range_mode,
            post_margin_bars=args.post_margin_bars,
            grid_dir=grid_dir,
            decode=DecodeConfig(
                temperature=0.0 if args.greedy else args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        )
        print(json.dumps({"charts": 1, "donor": donor_rel, "notes": stats.to_dict()}, indent=2))
        return

    paths = _teacher_paths(args)
    if not paths:
        raise SystemExit("no charts to evaluate")

    if args.mode == "generate":
        from audio2map.utils.paths import raw_dir

        root = raw_dir()
        modes = tuple(args.generation_mode or ("constrained",))
        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip()) or (0,)
        if "unconstrained" in modes:
            if any(scenario != "matched_full" for scenario in scenarios):
                raise SystemExit("unconstrained validity runs only for matched_full")
            if seeds != (0,):
                raise SystemExit("unconstrained validity runs only for generation seed 0")
        prepared = materialize_eval_charts(
            paths,
            rng=random.Random(args.seed),
            build_grid_if_missing=args.build_grid,
            fixed_start_bar=args.start_bar,
            grid_dir=grid_dir,
            raw_root=root,
            need_relpath=True,
        )
        try:
            assert_materialization_complete(paths, prepared)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not prepared:
            raise SystemExit("no charts with a condition vector and a teacher window")
        ready = [(chart.path, chart.meta, chart.cond_vec) for chart in prepared]
        rels = [chart.relpath for chart in prepared]
        pinned = None
        if official:
            try:
                pinned = pin_official_candidate_pool(
                    rels,
                    default_donor_map_path(),
                    seed=RANDOM_FULL_SEED,
                )
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
        donor = None
        if "random_full" in scenarios:
            if pinned is not None:
                donor = pinned
            else:
                map_path = donor_map or default_donor_map_path()
                try:
                    donor = require_donor_map(map_path, rels, seed=RANDOM_FULL_SEED)
                except Exception as exc:
                    raise SystemExit(str(exc)) from exc
        diversity_rels = None
        explicit_diversity_off = args.diversity_charts == 0 and not args.diversity_osu
        if not explicit_diversity_off and (args.diversity_charts or args.diversity_osu or official):
            explicit = (
                [chart_key(Path(osu), root) for osu in args.diversity_osu]
                if args.diversity_osu
                else None
            )
            diversity_count = (
                DIVERSITY_SUBSET_COUNT if args.diversity_charts is None else args.diversity_charts
            )
            try:
                subset = require_chart_subset(
                    Path(args.diversity_subset) if args.diversity_subset else default_diversity_subset_path(),
                    rels,
                    seed=DIVERSITY_SUBSET_SEED,
                    count=diversity_count,
                    kind="diversity subset",
                )
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            if explicit is not None and tuple(sorted(explicit)) != tuple(sorted(subset.charts)):
                raise SystemExit("explicit diversity charts do not match the frozen subset")
            diversity_rels = list(subset.charts)
            logger.info("diversity subset charts=%d", len(diversity_rels))
        unconstrained_rels = None
        if "unconstrained" in modes:
            try:
                unc = require_chart_subset(
                    Path(args.unconstrained_subset)
                    if args.unconstrained_subset
                    else default_unconstrained_subset_path(),
                    rels,
                    seed=UNCONSTRAINED_SUBSET_SEED,
                    count=UNCONSTRAINED_SUBSET_COUNT,
                    kind="unconstrained subset",
                )
            except Exception as exc:
                raise SystemExit(str(exc)) from exc
            unconstrained_rels = list(unc.charts)
            logger.info("unconstrained subset charts=%d", len(unconstrained_rels))
        motif_path = Path(args.motif_vocab) if args.motif_vocab else default_motif_lists_path()
        if not motif_path.is_file():
            raise SystemExit(f"motif lists missing: {motif_path}")
        motif_vocab = json.loads(motif_path.read_text(encoding="utf-8"))
        try:
            assert_motif_list_count(motif_vocab)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        report = eval_generation(
            model,
            ready,
            device=device,
            scenarios=scenarios,
            modes=modes,
            seeds=seeds,
            donor=donor,
            raw_root=root,
            diversity_rels=diversity_rels,
            unconstrained_rels=unconstrained_rels,
            motif_vocab=motif_vocab,
            decode=DecodeConfig(
                temperature=0.0 if args.greedy else args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
            build_grid_if_missing=args.build_grid,
            range_mode=args.range_mode,
            post_margin_bars=args.post_margin_bars,
            grid_dir=grid_dir,
        )
        print(json.dumps(report, indent=2))
        return

    logger.info("evaluating %d charts mode=%s", len(paths), args.mode)
    try:
        result = eval_teacher_forcing(
            model,
            paths,
            device=device,
            seed=args.seed,
            build_grid_if_missing=args.build_grid,
            fixed_start_bar=args.start_bar,
            grid_dir=grid_dir,
            scenarios=scenarios,
            donor_map_path=donor_map,
            donor_seed=RANDOM_FULL_SEED,
            official_exact=official,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if result.scenarios[scenarios[0]].charts == 0:
        raise SystemExit("could not build any eval windows")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
