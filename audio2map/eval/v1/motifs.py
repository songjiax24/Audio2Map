"""Adapt v1 rows to the shared motif fitter without millisecond or v0 tick snapping."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from audio2map.eval.motif import MotifSequences, motif_length
from audio2map.eval.motif_fit import evaluator_vocab, fit_scopes, scope_sequences
from audio2map.eval.motif_freq import OrderedMotifVocab, load_ordered_motif_vocab
from audio2map.grid.v1.positions import V80_DENS
from audio2map.tokens.v1.vocab import TOKEN_BEAT, TOKENIZER_VERSION, parse_pos_token
from audio2map.tokens.vocab import TOKEN_POS_PREFIX, TOKEN_ROW_PREFIX, row_state_from_token

FORMAT = "v1-grid-motifs-1"
DEFAULT_MOTIF_K = 1000
# Exact integer encoding of V80 fractions, not an additional quantization grid.
BEAT_UNITS = math.lcm(*V80_DENS)


def motif_sequences(tokens: Sequence[str]) -> MotifSequences:
    """Read validated token rows; every BEAT advances one BeatGrid beat index."""
    beat = 0
    position: int | None = None
    rows = {}
    for token in tokens:
        if token == TOKEN_BEAT:
            beat += 1
        elif token.startswith(TOKEN_POS_PREFIX):
            numer, denom = parse_pos_token(token)
            position = beat * BEAT_UNITS + numer * (BEAT_UNITS // denom)
        elif token.startswith(TOKEN_ROW_PREFIX):
            if position is None:
                raise ValueError("ROW without a POS in a v1 motif sequence")
            rows[position] = row_state_from_token(token)
            position = None
    return MotifSequences(rows)


class _PackedSymbols(Sequence):
    """Keep a fitting corpus compact; the shared fitter still reads ordinary symbols."""

    def __init__(self, symbols) -> None:
        self.table = tuple(dict.fromkeys(symbols))
        indices = {symbol: index for index, symbol in enumerate(self.table)}
        self.ids = np.fromiter((indices[symbol] for symbol in symbols), dtype=np.int16)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.table[int(self.ids[index])]


def fit_vocabulary(charts: Iterable[Sequence[str]], *, k: int, corpus: dict) -> dict:
    if corpus.get("split") != "train":
        raise ValueError("v1 motif vocabulary must be fitted on the train split")
    scopes = {scope: [] for scope in ("lane", "hand", "row")}
    for tokens in charts:
        for scope, sequences in scope_sequences(motif_sequences(tokens)).items():
            scopes[scope].extend(
                (_PackedSymbols(symbols), np.asarray(positions, dtype=np.int64))
                for symbols, positions in sequences
            )
    fitted = fit_scopes(scopes, k_fit=k)
    return {
        "format": FORMAT,
        "tokenizer": TOKENIZER_VERSION,
        "beat_units": BEAT_UNITS,
        "corpus": corpus,
        "k": k,
        "vocabulary": evaluator_vocab(fitted),
    }


def load_vocabulary(data: dict) -> OrderedMotifVocab:
    if (
        data.get("format") != FORMAT
        or data.get("tokenizer") != TOKENIZER_VERSION
        or data.get("beat_units") != BEAT_UNITS
        or data.get("corpus", {}).get("split") != "train"
    ):
        raise ValueError("expected a v1 grid motif vocabulary fitted on train")
    return load_ordered_motif_vocab(data["vocabulary"])


def window_counts(tokens: Sequence[str], grid, vocabulary: OrderedMotifVocab) -> dict:
    """Eligible spans and redline crossings for each frozen vocabulary length.

    This audits the current beat-index coordinates; it does not redefine gaps.
    """
    result = {}
    for scope, sequences in scope_sequences(motif_sequences(tokens)).items():
        lengths = sorted({motif_length(key) for key in vocabulary.keys[scope]})
        result[scope] = {}
        for length in lengths:
            possible = crossing = 0
            for _symbols, positions in sequences:
                count = max(0, len(positions) - length + 1)
                possible += count
                crossing += sum(
                    grid.beats[positions[i] // BEAT_UNITS].section
                    != grid.beats[positions[i + length - 1] // BEAT_UNITS].section
                    for i in range(count)
                )
            result[scope][str(length)] = {"windows": possible, "redline_crossing": crossing}
    return result
