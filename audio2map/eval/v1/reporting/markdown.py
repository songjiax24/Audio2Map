"""Render the main tables from a completed v1 evaluation summary."""

from __future__ import annotations

from audio2map.eval.v1.statistics import mean_available


def markdown(report: dict) -> str:
    def value(number):
        return "—" if number is None else f"{number:.4f}"

    run = report["run"]
    lines = [
        "# v1 evaluation",
        "",
        f"Checkpoint: `{run['checkpoint']}`",
        f"Split: {run['split']}; charts: {len(report['pool']['accepted'])}; "
        f"diagnostic subset: {run['subset']}; motif train subset: {run['motif_training_subset']}",
        f"Device: {run['gpu'] or run['device']}; elapsed: {report['elapsed_s']:.1f} s; "
        f"reused records: {report['reused_records']}",
        "",
    ]
    if report["teacher"]:
        lines.extend(
            [
                "## Teacher",
                "",
                "| Scenario | NLL/token | Perplexity | Accuracy | ROW accuracy | "
                "Legal probability |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in report["teacher"].items():
            lines.append(
                f"| {name} | {value(row['loss_token'])} | {value(row['perplexity'])} | "
                f"{value(row['acc_token'])} | {value(row['by_type']['row']['accuracy'])} | "
                f"{value(row['legal_prob_token'])} |"
            )
    if report["constrained"]:
        lines.extend(
            [
                "",
                "## Constrained generation",
                "",
                "| Scenario | Samples | Endpoint micro F1 | SR MAE | MSD overall MAE | "
                "Concurrent | Almost concurrent |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in report["constrained"].items():
            error = row["condition_error"]
            plausible = row["plausibility"]
            lines.append(
                f"| {name} | {row['samples']} | {value(row['match']['micro']['f1'])} | "
                f"{value(error['official_sr']['mae'])} | {value(error['msd_overall']['mae'])} | "
                f"{plausible['concurrent']} | {plausible['almost_concurrent']} |"
            )
        lines.extend(
            [
                "",
                "Plausibility counts sum over samples; per-chart counts and "
                "all 18 condition fields are in summary.json.",
            ]
        )
    if report["unconstrained"]:
        lines.extend(
            [
                "",
                "## Unconstrained validity",
                "",
                "| Scenario | Samples | Valid rate | Illegal token | Truncated |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, row in report["unconstrained"].items():
            lines.append(
                f"| {name} | {row['samples']} | {value(row['valid_rate'])} | "
                f"{row['counts'].get('illegal_token', 0)} | {row['counts'].get('truncated', 0)} |"
            )
    if report["diversity"]:
        pairs = [pair for chart in report["diversity"].values() for pair in chart["pairs"]]
        lines.extend(["", "## Diversity", "", f"Pairs: {len(pairs)}."])
        if pairs:
            lines.extend(
                [
                    f"Mean pairwise endpoint F1: {value(mean_available(pair['f1'] for pair in pairs))}.",
                    f"Identical pair rate: {value(mean_available(pair['identical'] for pair in pairs))}.",
                ]
            )
    return "\n".join(lines) + "\n"
