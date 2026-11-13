"""Shared whole-chart validation for training and the v1 evaluation CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from audio2map.model.v1.batch import V1Batch
from audio2map.model.v1.loss import accuracy_counts, ce_by_type, legal_tally, nonpad_count, plain_ce
from audio2map.model.v1.model import V1ChartModel
from audio2map.metrics.token import SplitTokenAccuracy, split_token_accuracy
from audio2map.tokens.v1.vocab import TOKEN_PAD, build_vocab, invert_vocab


@dataclass(frozen=True, slots=True)
class ValidationReadout:
    loss_chart: float
    loss_token: float
    acc_chart: float
    acc_token: float
    pos: float
    row: float
    beat: float
    eos: float
    legal_top1_chart: float = 0.0
    legal_top1_token: float = 0.0
    legal_prob_chart: float = 0.0
    legal_prob_token: float = 0.0
    legal_top1_pos: float = 0.0
    legal_top1_row: float = 0.0
    legal_top1_beat: float = 0.0
    legal_top1_eos: float = 0.0
    legal_prob_pos: float = 0.0
    legal_prob_row: float = 0.0
    legal_prob_beat: float = 0.0
    legal_prob_eos: float = 0.0
    loss_pos: float = 0.0
    loss_row: float = 0.0
    loss_beat: float = 0.0
    loss_eos: float = 0.0
    row_counts: dict[str, tuple[int, int]] | None = None


@torch.no_grad()
def validate_charts(
    model: V1ChartModel,
    charts: list[V1Batch],
    device: torch.device,
    *,
    use_amp: bool,
    masks: list[torch.Tensor] | None = None,
) -> ValidationReadout:
    """Per-chart means, then the token-weighted loss and accuracy.

    ``masks`` replaces ``cond_m`` and leaves the cached audio and tokens.
    """
    model.eval()
    chart_total = 0.0
    token_total = 0.0
    token_count = 0
    seen = 0
    chart_acc = 0.0
    chart_acc_n = 0
    correct = {name: 0 for name in ("all", "pos", "row", "beat", "eos")}
    counted = {name: 0 for name in correct}
    legal_hits = 0
    legal_mass = 0.0
    legal_count = 0
    legal_chart_top1 = 0.0
    legal_chart_prob = 0.0
    legal_charts = 0
    token_types = ("pos", "row", "beat", "eos")
    legal_type_hits = {name: 0 for name in token_types}
    legal_type_mass = {name: 0.0 for name in token_types}
    legal_type_count = {name: 0 for name in token_types}
    loss_type_sum = {name: 0.0 for name in token_types}
    loss_type_count = {name: 0 for name in token_types}
    row_details = SplitTokenAccuracy()
    names = invert_vocab()
    pad_id = build_vocab()[TOKEN_PAD]
    for index, example in enumerate(charts):
        if masks is not None:
            example = replace(example, cond_m=masks[index])
        batch = to_device(example, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch)
            targets = batch.token_ids[:, 1:]
            loss = plain_ce(logits, targets)
        count = nonpad_count(targets)
        value = float(loss.detach().item())
        chart_total += value
        token_total += value * count
        token_count += count
        seen += 1
        counts = accuracy_counts(logits, targets)
        row_details.merge(split_token_accuracy(
            logits.argmax(dim=-1)[0], targets[0], targets[0].ne(pad_id), id_to_token=names,
        ))
        hits, total = counts["all"]
        if total:
            chart_acc += hits / total
            chart_acc_n += 1
        for name, (name_hits, name_total) in counts.items():
            correct[name] += name_hits
            counted[name] += name_total
        for name, (type_sum, type_n) in ce_by_type(logits, targets).items():
            loss_type_sum[name] += type_sum
            loss_type_count[name] += type_n
        if example.legal_ids is None:
            raise RuntimeError("validation chart has no legal-id rows")
        tally = legal_tally(logits, targets, example.legal_ids)
        legal_hits += tally.hits
        legal_mass += tally.mass
        legal_count += tally.count
        if tally.count:
            legal_chart_top1 += tally.hits / tally.count
            legal_chart_prob += tally.mass / tally.count
            legal_charts += 1
        for name, (type_hits, type_mass, type_n) in tally.by_type.items():
            legal_type_hits[name] += type_hits
            legal_type_mass[name] += type_mass
            legal_type_count[name] += type_n
    if seen == 0 or token_count == 0:
        raise RuntimeError("validation produced no charts")

    def ratio(name: str) -> float:
        total = counted[name]
        return correct[name] / total if total else 0.0

    def loss_type(name: str) -> float:
        total = loss_type_count[name]
        return loss_type_sum[name] / total if total else 0.0

    def legal_type(name: str, *, mass: bool) -> float:
        total = legal_type_count[name]
        if not total:
            return 0.0
        value = legal_type_mass[name] if mass else legal_type_hits[name]
        return value / total

    return ValidationReadout(
        loss_chart=chart_total / seen,
        loss_token=token_total / token_count,
        acc_chart=chart_acc / chart_acc_n if chart_acc_n else 0.0,
        acc_token=ratio("all"),
        pos=ratio("pos"),
        row=ratio("row"),
        beat=ratio("beat"),
        eos=ratio("eos"),
        legal_top1_chart=(legal_chart_top1 / legal_charts if legal_charts else 0.0),
        legal_top1_token=legal_hits / legal_count if legal_count else 0.0,
        legal_prob_chart=(legal_chart_prob / legal_charts if legal_charts else 0.0),
        legal_prob_token=legal_mass / legal_count if legal_count else 0.0,
        legal_top1_pos=legal_type("pos", mass=False),
        legal_top1_row=legal_type("row", mass=False),
        legal_top1_beat=legal_type("beat", mass=False),
        legal_top1_eos=legal_type("eos", mass=False),
        legal_prob_pos=legal_type("pos", mass=True),
        legal_prob_row=legal_type("row", mass=True),
        legal_prob_beat=legal_type("beat", mass=True),
        legal_prob_eos=legal_type("eos", mass=True),
        loss_pos=loss_type("pos"),
        loss_row=loss_type("row"),
        loss_beat=loss_type("beat"),
        loss_eos=loss_type("eos"),
        row_counts={
            "row_lane": (row_details.row_lane_correct, row_details.row_lane_total),
            "tap_recall": (row_details.tap_recall_num, row_details.tap_recall_den),
            "hold_head_recall": (
                row_details.hold_start_recall_num, row_details.hold_start_recall_den,
            ),
            "hold_tail_recall": (row_details.hold_end_recall_num, row_details.hold_end_recall_den),
        },
    )


def to_device(batch: V1Batch, device: torch.device) -> V1Batch:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.to(device, non_blocking=True)

    return V1Batch(
        frame_lo=move(batch.frame_lo),
        cond_x=move(batch.cond_x),
        cond_m=move(batch.cond_m),
        token_ids=move(batch.token_ids),
        cursor_time_ms=move(batch.cursor_time_ms),
        phi=move(batch.phi),
        beat_b=move(batch.beat_b),
        log_bpm=move(batch.log_bpm),
        rel_u=move(batch.rel_u),
        coverage_start_ms=move(batch.coverage_start_ms),
        coverage_end_ms=move(batch.coverage_end_ms),
        redline_offset_ms=move(batch.redline_offset_ms),
        redline_beat_ms=move(batch.redline_beat_ms),
        audio_mask=move(batch.audio_mask),
        token_mask=move(batch.token_mask),
        mel=move(batch.mel),
        n_frames=move(batch.n_frames),
        legal_ids=batch.legal_ids,
    )
