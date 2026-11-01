"""The v1 gate drops parse failures, empty charts, and a spliced timeline."""

from __future__ import annotations

from types import SimpleNamespace

from audio2map.dataset.split import ChartSnapshot
from audio2map.dataset.v1.gate import (
    INVALID_HIT_OBJECTS,
    NO_NOTES,
    SPLICED_TIMELINE,
    chart_reject_reason,
    filter_v1_charts,
    osu_reject_reason,
    structure_reject_reason,
)
from audio2map.utils.errors import ENDPOINT_OUTSIDE_AUDIO

_HEADER = (
    "osu file format v14\n"
    "[General]\nMode:3\nAudioFilename:a.wav\n"
    "[Difficulty]\nCircleSize:4\n"
    "[TimingPoints]\n0,500,4,2,0,50,1,0\n"
)


def _write(path, body: str) -> None:
    path.write_text(_HEADER + body, encoding="utf-8")


def _chart(relpath: str) -> ChartSnapshot:
    return ChartSnapshot(relpath, "set", "train")


def _row(**kwargs):
    base = dict(error=None, n_frames=8, n_tokens=4, t_audio_ms=10_000.0, relpath="set/a.osu")
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_meter_is_not_a_reason(tmp_path) -> None:
    path = tmp_path / "meter.osu"
    path.write_text(
        "osu file format v14\n"
        "[General]\nMode:3\n"
        "[Difficulty]\nCircleSize:4\n"
        "[TimingPoints]\n0,500,6,2,0,50,1,0\n"
        "[HitObjects]\n64,192,0,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )
    assert osu_reject_reason(path) is None


def test_parse_and_empty_charts_are_dropped(tmp_path) -> None:
    empty = tmp_path / "empty.osu"
    _write(empty, "[HitObjects]\n")
    hold = tmp_path / "hold.osu"
    _write(hold, "[HitObjects]\n64,192,100,128,0,100:0:0:0:\n")
    late = tmp_path / "late.osu"
    _write(late, "[HitObjects]\n64,192,5000,1,0,0:0:0:0:\n")
    assert osu_reject_reason(empty) == NO_NOTES
    assert osu_reject_reason(hold) == INVALID_HIT_OBJECTS
    assert structure_reject_reason(late, 1000.0) == ENDPOINT_OUTSIDE_AUDIO


def test_spliced_log_drops_and_a_second_call_uses_the_cache(tmp_path, monkeypatch) -> None:
    root = tmp_path / "raw"
    folder = root / "set"
    folder.mkdir(parents=True)
    _write(folder / "a.osu", "[HitObjects]\n64,192,0,1,0,0:0:0:0:\n")
    (folder / "a.wav").write_bytes(b"pcm")
    calls = {"n": 0}

    def log(_path, **_kwargs):
        calls["n"] += 1
        return (
            "Error submitting packet to decoder: Invalid data found when processing input\n"
            "timestamp discontinuity of 448 samples\n"
        )

    monkeypatch.setattr("audio2map.dataset.v1.gate.ffmpeg_decode_log", log)
    charts = [_chart("set/a.osu")]
    rows = [_row()]
    gate = tmp_path / "gate"
    kept, kept_rows = filter_v1_charts(
        charts, root, lengths=rows, workers=1, gate_dir=gate
    )
    assert kept == []
    assert kept_rows == []
    assert calls["n"] == 1
    again, _ = filter_v1_charts(charts, root, lengths=rows, workers=1, gate_dir=gate)
    assert again == []
    assert calls["n"] == 1


def test_packet_without_a_jump_stays(tmp_path, monkeypatch) -> None:
    root = tmp_path / "raw"
    folder = root / "set"
    folder.mkdir(parents=True)
    path = folder / "a.osu"
    _write(path, "[HitObjects]\n64,192,0,1,0,0:0:0:0:\n")
    (folder / "a.wav").write_bytes(b"pcm")
    monkeypatch.setattr(
        "audio2map.dataset.v1.gate.ffmpeg_decode_log",
        lambda _path, **_kwargs: "Header missing\n",
    )
    kept, kept_rows = filter_v1_charts(
        [_chart("set/a.osu")],
        root,
        lengths=[_row()],
        workers=1,
        gate_dir=tmp_path / "gate",
    )
    assert [chart.relpath for chart in kept] == ["set/a.osu"]
    assert len(kept_rows) == 1
    assert (
        chart_reject_reason(
            path,
            t_audio_ms=10_000.0,
            ffmpeg_log="Header missing\n",
            structure_known_ok=True,
        )
        is None
    )
    assert (
        chart_reject_reason(
            path,
            t_audio_ms=10_000.0,
            ffmpeg_log="Header missing\ntimestamp discontinuity\n",
            structure_known_ok=True,
        )
        == SPLICED_TIMELINE
    )
