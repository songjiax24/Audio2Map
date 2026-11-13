import pytest

from audio2map.eval.motif_fit import evaluator_vocab, fit_scopes, scope_sequences
from audio2map.eval.motif_freq import MotifCorpus, corpus_distance
from audio2map.eval.v1.motifs import BEAT_UNITS, fit_vocabulary, load_vocabulary, motif_sequences
from audio2map.eval.v1.motifs import window_counts
from audio2map.osu.schema import ManiaNote, NoteType
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import encode_chart


TOKENS = [
    "<BOS>",
    "<POS_0/1>",
    "<ROW_1000>",
    "<POS_1/7>",
    "<ROW_0100>",
    "<BEAT>",
    "<POS_1/32>",
    "<ROW_1100>",
    "<BEAT>",
    "<EOS>",
]


def test_exact_v80_positions_and_hand_mirroring():
    sequences = motif_sequences(TOKENS)
    assert [position for position, _ in sequences.row] == [
        0,
        BEAT_UNITS // 7,
        BEAT_UNITS + BEAT_UNITS // 32,
    ]
    assert len(scope_sequences(sequences)["lane"]) == 2


def test_v1_vocabulary_roundtrip_and_shared_frequency_distance():
    data = fit_vocabulary([TOKENS], k=8, corpus={"split": "train"})
    vocabulary = load_vocabulary(data)
    human = MotifCorpus(vocabulary)
    generated = MotifCorpus(vocabulary)
    human.add_sequences(motif_sequences(TOKENS))
    generated.add_sequences(motif_sequences(TOKENS))
    result = corpus_distance(generated, human)
    assert all(result[scope]["mae"] == 0 for scope in ("lane", "hand", "row"))
    assert all(sum(result[scope]["length_contributions"].values()) == 0 for scope in result)
    with pytest.raises(ValueError):
        load_vocabulary(data["vocabulary"])
    with pytest.raises(ValueError):
        fit_vocabulary([TOKENS], k=8, corpus={"split": "test"})


def test_packed_training_corpus_preserves_the_shared_fit_exactly():
    shifted = [token.replace("<POS_1/7>", "<POS_1/32>") for token in TOKENS]
    charts = [TOKENS, shifted]
    scopes = {name: [] for name in ("lane", "hand", "row")}
    for tokens in charts:
        for name, sequences in scope_sequences(motif_sequences(tokens)).items():
            scopes[name].extend(sequences)
    expected = evaluator_vocab(fit_scopes(scopes, k_fit=8))
    fitted = fit_vocabulary(iter(charts), k=8, corpus={"split": "train"})
    assert fitted["vocabulary"] == expected


def test_motif_error_contributions_and_redline_exposure_are_explicit():
    chart = encode_chart(
        [ManiaNote(t, 0, NoteType.TAP) for t in (0, 500, 750, 1000)],
        [Redline(0, 500, 4, 0), Redline(750, 500, 4, 1)], 1500,
    )
    fitted = fit_vocabulary([chart.tokens], k=20, corpus={"split": "train"})
    vocab = load_vocabulary(fitted)
    human, generated = MotifCorpus(vocab), MotifCorpus(vocab)
    human.add_sequences(motif_sequences(chart.tokens))
    generated.add_sequences(motif_sequences(TOKENS))
    for scope in corpus_distance(generated, human).values():
        assert sum(scope["length_contributions"].values()) == pytest.approx(scope["mae"])
        assert sum(row["mae_contribution"] for row in scope["by_length"].values()) == pytest.approx(
            scope["mae"]
        )
    windows = window_counts(chart.tokens, chart.grid, vocab)
    assert windows["lane"]["1"]["redline_crossing"] == 0
    assert windows["lane"]["2"] == {"windows": 3, "redline_crossing": 1}
