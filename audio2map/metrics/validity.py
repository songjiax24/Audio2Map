"""Teacher-forcing logit-level legality (not a loss)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from audio2map.tokens import TOKEN_ROW_PREFIX, ChartState, build_vocab, invert_vocab, row_state_from_token


@dataclass(slots=True)
class ValidityStats:
    top1_count: float = 0.0
    prob_sum: float = 0.0
    n_valid: float = 0.0

    def merge(self, other: ValidityStats) -> None:
        self.top1_count += other.top1_count
        self.prob_sum += other.prob_sum
        self.n_valid += other.n_valid

    @property
    def legal_top1(self) -> float:
        return self.top1_count / self.n_valid if self.n_valid else 0.0

    @property
    def legal_probability(self) -> float:
        return self.prob_sum / self.n_valid if self.n_valid else 0.0


def logit_validity(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    window_bars: int,
    id_to_token: dict[int, str] | None = None,
    vocab: dict[str, int] | None = None,
) -> ValidityStats:
    """Micro-average legal top-1 and legal probability over masked positions.

    ``logits[b, t]`` predicts ``token_ids[b, t + 1]``. Positions with
    ``loss_mask[b, t + 1] == 0`` are skipped (same mask as CE / token acc).
    Window ``<EOS>`` with active holds is legal.
    """
    vocab = vocab or build_vocab()
    id_to_token = id_to_token or invert_vocab(vocab)
    stats = ValidityStats()
    logits_f = logits.float()
    for b in range(token_ids.shape[0]):
        ids = token_ids[b]
        mask = loss_mask[b]
        seq_logits = logits_f[b]
        if ids.shape[0] < 3:
            continue
        init_tok = id_to_token[int(ids[1].item())]
        if not init_tok.startswith(TOKEN_ROW_PREFIX):
            continue
        state = ChartState.from_initial_row(
            row_state_from_token(init_tok),
            window_bars=window_bars,
            vocab=vocab,
        )
        for t in range(1, ids.shape[0] - 1):
            next_id = int(ids[t + 1].item())
            if float(mask[t + 1].item()) == 0.0:
                continue
            allowed = state.allowed_token_ids()
            if not allowed:
                raise RuntimeError(
                    "empty legal token set at a masked position "
                    f"(batch={b}, t={t}, bars_done={state.bars_done}, "
                    f"in_bar={state.in_bar}, expect_row={state.expect_row}, "
                    f"finished={state.finished})"
                )
            row_logits = seq_logits[t]
            pred = int(row_logits.argmax().item())
            stats.top1_count += float(pred in allowed)
            probs = F.softmax(row_logits, dim=-1)
            idx = torch.tensor(sorted(allowed), device=probs.device, dtype=torch.long)
            stats.prob_sum += float(probs[idx].sum().item())
            stats.n_valid += 1.0
            state.observe(next_id)
    return stats
