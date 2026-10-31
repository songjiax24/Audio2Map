"""Constrained samples stay legal, and the .osu keeps awkward notes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audio2map.features.audio.v1.constants import SAMPLE_RATE
from audio2map.features.cond.chart import ChartMeta
from audio2map.features.cond.v1.vec import COND_VEC_DIM, COND_VEC_VERSION
from audio2map.generate.v1.chart import condition_from_json, generate_v1_chart
from audio2map.generate.v1.notes import hit_lines_from_events, hit_lines_from_notes
from audio2map.generate.v1 import sample as sample_mod
from audio2map.generate.v1.sample import _choose, sample_chart
from audio2map.generate.v1.stream import encode_context
from audio2map.grid.v1.beatgrid import build_beatgrid
from audio2map.model.v1.model import V1ChartModel
from audio2map.osu.parser import parse_beatmap
from audio2map.osu.v1.export import write_v1_osu
from audio2map.osu.v1.timing import Redline
from audio2map.tokens.v1.codec import DecodedEvent, decode_tokens
from audio2map.tokens.v1.structure import EndpointKind
from audio2map.tokens.v1.vocab import build_vocab, invert_vocab

_SOURCE = """osu file format v14

[General]
AudioFilename: audio.wav
Mode: 3

[Metadata]
Title:Tiny
Artist:Test
Creator:Me
Version:Easy
BeatmapID:12

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,500,4,2,0,100,1,0
100,-50,4,2,0,100,0,0

[HitObjects]
64,192,0,1,0,0:0:0:0:
"""


def test_illegal_mode_is_not_sampled() -> None:
    vocab = build_vocab()
    names = invert_vocab(vocab)
    logits = torch.zeros(len(vocab))
    logits[vocab["<BOS>"]] = 10.0
    logits[vocab["<BEAT>"]] = 1.0
    chosen = _choose(logits, {"<BEAT>"}, vocab, names, 0.0, torch.Generator())
    assert chosen == "<BEAT>"


def test_short_sample_decodes_and_exports(tmp_path: Path, monkeypatch) -> None:
    torch.manual_seed(0)
    model = V1ChartModel().eval()
    seconds = 1.2
    pcm = np.random.default_rng(0).standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32)
    pcm *= 0.05
    redlines = (Redline(0.0, 500.0, 4, 0),)
    grid = build_beatgrid(redlines, seconds * 1000.0)
    context, g, _segments = encode_context(model, pcm, grid, redlines, device=torch.device("cpu"))
    cond = model.cond(torch.zeros(1, COND_VEC_DIM), torch.zeros(1, COND_VEC_DIM))
    calls = {"n": 0}
    real_legal = sample_mod.legal_tokens

    def _count(cursor, chart):
        calls["n"] += 1
        return real_legal(cursor, chart)

    monkeypatch.setattr(sample_mod, "legal_tokens", _count)
    monkeypatch.setattr("audio2map.tokens.v1.state.legal_tokens", _count)
    tokens = sample_chart(model, context, g, grid, cond=cond, seed=0, temperature=0.0)
    assert calls["n"] == len(tokens) - 1
    assert tokens[0] == "<BOS>"
    assert tokens[-1] == "<EOS>"
    events = decode_tokens(tokens, grid)
    source = tmp_path / "in.osu"
    source.write_text(_SOURCE, encoding="utf-8")
    out = tmp_path / "out.osu"
    hit_lines, warnings = hit_lines_from_events(events)
    audit = write_v1_osu(out, hit_lines, warnings, redlines, source)
    text = audit.path.read_text(encoding="utf-8")
    assert "100,-50," not in text
    assert "\n0,500,4,2,0,100,1,0\n" in text
    assert "Mode:3" in text
    assert "OverallDifficulty:8" in text
    assert "Version:Easy Audio2Map" in text
    parsed = parse_beatmap(audit.path)
    assert parsed.metadata.title == "Tiny"
    assert all(point.uninherited for point in parsed.timing_points)


def test_generate_returns_notes_and_the_caller_writes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = V1ChartModel().eval()
    seconds = 1.2
    pcm = np.random.default_rng(1).standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32)
    pcm *= 0.05
    redlines = (Redline(0.0, 500.0, 4, 0),)
    meta = ChartMeta(osu_path="", official_sr=4.5)
    chart = generate_v1_chart(
        model,
        pcm,
        redlines,
        meta,
        device=torch.device("cpu"),
        temperature=0.0,
    )
    assert chart.tokens[0] == "<BOS>"
    assert chart.tokens[-1] == "<EOS>"
    assert chart.frames > 0
    out = tmp_path / "made.osu"
    audit = write_v1_osu(
        out,
        hit_lines_from_notes(chart.notes),
        list(chart.warnings),
        redlines,
        audio_filename="song.mp3",
        title="Made",
        artist="Tester",
    )
    text = audit.path.read_text(encoding="utf-8")
    assert "\n0,500,4,2,0,100,1,0\n" in text
    assert "-50," not in text
    assert "AudioFilename: song.mp3" in text
    assert "AudioLeadIn: 0" in text
    assert "HPDrainRate:8" in text
    assert "Version:Audio2Map" in text
    parsed = parse_beatmap(audit.path)
    assert parsed.metadata.title == "Made"
    assert parsed.metadata.artist == "Tester"
    assert all(point.uninherited for point in parsed.timing_points)


def test_condition_json_needs_no_chart_path(tmp_path: Path) -> None:
    path = tmp_path / "cond.json"
    path.write_text(
        '{"cond_vec_version": %d, "official_sr": 4.5}\n' % COND_VEC_VERSION,
        encoding="utf-8",
    )
    meta = condition_from_json(path)
    assert meta.osu_path == ""
    assert meta.official_sr == 4.5


def test_export_keeps_a_shut_hold_and_a_duplicate(tmp_path: Path) -> None:
    events = [
        DecodedEvent(0, EndpointKind.TAP, 0, 0, 0, 1, 10.2, 0),
        DecodedEvent(0, EndpointKind.TAP, 0, 0, 1, 2, 10.8, 0),
        DecodedEvent(1, EndpointKind.HEAD, 0, 0, 0, 1, 0.0, 0),
        DecodedEvent(1, EndpointKind.TAIL, 0, 0, 1, 4, 0.4, 0),
        DecodedEvent(2, EndpointKind.HEAD, 0, 1, 0, 1, 50.0, 1),
    ]
    lines, warnings = hit_lines_from_events(events)
    assert any("two events at 10 ms" in item for item in warnings)
    assert any("not after head" in item for item in warnings)
    assert any("unclosed" in item for item in warnings)
    assert sum(line.startswith("64,192,10,1,") for line in lines) == 2
    times = [int(line.split(",")[2]) for line in lines]
    assert times == sorted(times)
    assert any(line.startswith("192,192,0,128,0,0:") for line in lines)
    assert any(line.startswith("320,192,50,128,0,50:") for line in lines)
    source = tmp_path / "in.osu"
    source.write_text(_SOURCE, encoding="utf-8")
    out = tmp_path / "awkward.osu"
    write_v1_osu(out, lines, warnings, (Redline(0.0, 500.0, 4, 0),), source)
    body = out.read_text(encoding="utf-8")
    assert body.count("64,192,10,1,") == 2
    assert "128,0,0:" in body
