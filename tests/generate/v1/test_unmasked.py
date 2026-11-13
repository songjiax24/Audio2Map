from types import SimpleNamespace

import torch

from audio2map.generate.v1 import sample as sampling
from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.vocab import build_vocab


def setup(monkeypatch, preferred):
    vocab = build_vocab()
    model = SimpleNamespace(training=False, decoder=[None])
    grid = build_beatgrid([Redline(0.0, 500.0, 4, 0)], 100.0)

    class Window:
        def __init__(self, *args):
            pass

        def prepare(self, *args):
            pass

    def logits(_model, _token, position, *args, **kwargs):
        values = torch.zeros(1, 1, len(vocab))
        values[0, 0, vocab[preferred[position]]] = 10
        values[0, 0, vocab["<BEAT>"]] += 2
        values[0, 0, vocab["<EOS>"]] += 1
        return values

    monkeypatch.setattr(sampling, "CrossWindow", Window)
    monkeypatch.setattr(sampling, "cached_logits", logits)
    return model, torch.zeros(1, 2, 512), torch.arange(2).unsqueeze(0), grid


def test_full_vocabulary_sampling_reports_the_first_illegal_token(monkeypatch):
    args = setup(monkeypatch, ["<PAD>", "<PAD>"])
    result = sampling.sample_unconstrained(
        *args,
        cond=torch.zeros(1, 256),
        seed=0,
        temperature=0,
    )
    assert result.validity == sampling.ILLEGAL_TOKEN
    assert result.position == 0
    assert result.illegal_token == "<PAD>"
    assert result.tokens == ("<BOS>", "<PAD>")
    constrained = sampling.sample_chart(
        *args,
        cond=torch.zeros(1, 256),
        seed=0,
        temperature=0,
    )
    assert constrained == ["<BOS>", "<BEAT>", "<EOS>"]


def test_unmasked_sampler_can_finish_and_reports_truncation(monkeypatch):
    args = setup(monkeypatch, ["<BEAT>", "<EOS>"])
    result = sampling.sample_unconstrained(
        *args,
        cond=torch.zeros(1, 256),
        seed=0,
        temperature=0,
    )
    assert result.validity == sampling.VALID
    assert result.tokens == ("<BOS>", "<BEAT>", "<EOS>")
    monkeypatch.setattr(sampling, "_token_limit", lambda _grid: 0)
    truncated = sampling.sample_unconstrained(
        *args,
        cond=torch.zeros(1, 256),
        seed=0,
        temperature=0,
    )
    assert truncated.validity == sampling.TRUNCATED
