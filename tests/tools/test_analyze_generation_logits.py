"""Resume checks for the native logits capture command."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from audio2map.eval.v1.inputs import file_digest
from audio2map.features.cond.chart import ChartMeta
from audio2map.utils.time import SAMPLE_RATE
from tools import analyze_generation_logits as diagnostic


@pytest.mark.parametrize("artifact", ["tokens.json", "chart.osu"])
def test_resume_rebuilds_incomplete_trace_case(tmp_path, monkeypatch, artifact):
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    reference = tmp_path / "reference.osu"
    reference.write_text("[TimingPoints]\n0,500,4,2,0,100,1,0\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"rows": [{
        "key": "case", "audio": str(audio), "audio_sha256": file_digest(audio),
        "reference": str(reference), "reference_sha256": file_digest(reference),
        "condition": ChartMeta(str(reference)).to_dict(), "seed": 0, "temperature": 1,
        "silent": True,
    }]}))
    out = tmp_path / "out"
    args = SimpleNamespace(manifest=manifest, checkpoint=checkpoint, output=out, device="cpu")
    calls = []

    def generate(*args, observer, **kwargs):
        calls.append(1)
        return SimpleNamespace(tokens=["<BOS>", "<EOS>"], notes=[], warnings=[])

    monkeypatch.setattr(diagnostic, "load_v1_model", lambda *_: object())
    monkeypatch.setattr(diagnostic, "generate_v1_chart", generate)
    monkeypatch.setattr(
        diagnostic, "LogitRecorder",
        lambda: SimpleNamespace(rows=[], save=lambda path: path.write_bytes(b"trace")),
    )
    diagnostic.run(args)
    diagnostic.run(args)
    assert len(calls) == 1
    (out / "case" / artifact).unlink()
    diagnostic.run(args)
    assert len(calls) == 2
    assert (out / "case" / artifact).is_file()
    assert "AudioFilename: silence.wav" in (out / "case" / "chart.osu").read_text()
    pcm, rate = sf.read(out / "case" / "silence.wav", dtype="float32")
    assert rate == SAMPLE_RATE and len(pcm) == SAMPLE_RATE and not np.any(pcm)
