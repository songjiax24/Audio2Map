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
    load_or_create_chart_subset,
)
from audio2map.dataset.donor import (
    chart_key,
    default_donor_map_path,
    load_donor_map,
    load_or_create_donor_map,
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
    eval_generation,
    eval_inference_chart,
    eval_teacher_forcing,
    materialize_eval_charts,
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
        help="target→donor JSON (default: DATA/splits/random_full.json when random_full is used)",
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
        help="generate: comma-separated seeds. Statistics stay per seed; diversity compares them",
    )
    p.add_argument(
        "--diversity-charts",
        type=int,
        default=0,
        help="generate: how many charts to pin for multi-seed note F1 (drawn from a sorted pool, not file order)",
    )
    p.add_argument(
        "--diversity-seed",
        type=int,
        default=0,
        help="generate: seed used only when creating the diversity subset file",
    )
    p.add_argument(
        "--diversity-osu",
        action="append",
        default=None,
        help="generate: explicit diversity chart (repeatable). Saved on first use; later runs must match the file",
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
        help="generate: JSON {lane, hand, row} lists of motif strings",
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
            if not map_path.is_file():
                raise SystemExit(f"infer random_full needs an existing donor map: {map_path}")
            mapping = load_donor_map(map_path)
            rel = chart_key(path, root)
            if rel not in mapping.pairs:
                raise SystemExit(f"{rel} is missing from donor map {map_path}")
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
        prepared = materialize_eval_charts(
            paths,
            rng=random.Random(args.seed),
            build_grid_if_missing=args.build_grid,
            fixed_start_bar=args.start_bar,
            grid_dir=grid_dir,
            raw_root=root,
            need_relpath=True,
        )
        if not prepared:
            raise SystemExit("no charts with a condition vector and a teacher window")
        ready = [(chart.path, chart.meta, chart.cond_vec) for chart in prepared]
        rels = [chart.relpath for chart in prepared]
        donor = None
        if "random_full" in scenarios:
            donor = load_or_create_donor_map(
                donor_map or default_donor_map_path(),
                rels,
                seed=args.seed,
            )
        diversity_rels = None
        if args.diversity_charts or args.diversity_osu:
            explicit = (
                [chart_key(Path(osu), root) for osu in args.diversity_osu]
                if args.diversity_osu
                else None
            )
            subset = load_or_create_chart_subset(
                Path(args.diversity_subset) if args.diversity_subset else default_diversity_subset_path(),
                rels,
                count=args.diversity_charts,
                seed=args.diversity_seed,
                explicit=explicit,
            )
            diversity_rels = list(subset.charts)
            logger.info("diversity subset charts=%d", len(diversity_rels))
        motif_vocab = None
        if args.motif_vocab:
            motif_vocab = json.loads(Path(args.motif_vocab).read_text(encoding="utf-8"))
        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
        report = eval_generation(
            model,
            ready,
            device=device,
            scenarios=scenarios,
            modes=tuple(args.generation_mode or ("constrained",)),
            seeds=seeds or (0,),
            donor=donor,
            raw_root=root,
            diversity_rels=diversity_rels,
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
    )
    if result.scenarios[scenarios[0]].charts == 0:
        raise SystemExit("could not build any eval windows")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
