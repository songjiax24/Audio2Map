"""Tests for scripts.collect (Sayobot collector) — network fully mocked."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.collect import state as collector_state
from scripts.collect.sayobot import (
    _safe_extract,
    apply_4k_filter,
    audio_filename,
    chart_audio_map,
    fetch_page,
    info_has_mania_4k,
    is_mania_4k,
    layout_ok,
    prepare_set,
    prune,
    run,
)

def _mania_osu(audio: str = "song.mp3") -> str:
    return (
        "osu file format v14\n\n[General]\n"
        f"AudioFilename:{audio}\nMode:3\n\n[Difficulty]\nCircleSize:4\n"
    )


_MANIA_4K = _mania_osu()

_STD = """\
osu file format v14

[General]
AudioFilename:song.mp3
Mode:0

[Difficulty]
CircleSize:4
"""


def _write_set(tmp_path: Path, *, sid: str = "123") -> Path:
    set_dir = tmp_path / sid
    set_dir.mkdir()
    (set_dir / "song.mp3").write_bytes(b"\0" * 300_000)
    return set_dir


def test_is_mania_4k(tmp_path: Path) -> None:
    mania = tmp_path / "mania.osu"
    mania.write_text(_MANIA_4K, encoding="utf-8")
    std = tmp_path / "std.osu"
    std.write_text(_STD, encoding="utf-8")
    assert is_mania_4k(mania)
    assert not is_mania_4k(std)
    assert audio_filename(mania) == "song.mp3"


def test_chart_audio_map_follows_osu_filename(tmp_path: Path) -> None:
    set_dir = tmp_path / "1"
    set_dir.mkdir()
    (set_dir / "song.mp3").write_bytes(b"\0" * 1000)
    (set_dir / "drum-hitnormal.mp3").write_bytes(b"\0" * 300_000)
    osu = set_dir / "a.osu"
    osu.write_text(_MANIA_4K, encoding="utf-8")
    mapping = chart_audio_map(set_dir, [osu])
    assert mapping is not None and mapping[osu].name == "song.mp3"

    missing = tmp_path / "2"
    missing.mkdir()
    (missing / "other.mp3").write_bytes(b"\0" * 300_000)
    osu2 = missing / "a.osu"
    osu2.write_text(_MANIA_4K, encoding="utf-8")
    assert chart_audio_map(missing, [osu2]) is None


def test_layout_ok_and_chart_audio(tmp_path: Path) -> None:
    set_dir = _write_set(tmp_path)
    (set_dir / "a.osu").write_text(_MANIA_4K, encoding="utf-8")
    assert layout_ok(set_dir)

    mapping = chart_audio_map(set_dir, [set_dir / "a.osu"])
    assert mapping is not None and mapping[set_dir / "a.osu"].name == "song.mp3"

    extra = set_dir / "bg.jpg"
    extra.write_bytes(b"x")
    assert not layout_ok(set_dir)
    extra.unlink()

    orphan = set_dir / "orphan.mp3"
    orphan.write_bytes(b"x")
    assert not layout_ok(set_dir)
    orphan.unlink()

    sub = set_dir / "sub"
    sub.mkdir()
    assert not layout_ok(set_dir)


def test_prune_keeps_per_chart_audio(tmp_path: Path) -> None:
    set_dir = tmp_path / "9"
    set_dir.mkdir()
    (set_dir / "a.ogg").write_bytes(b"a" * 1000)
    (set_dir / "b.mp3").write_bytes(b"b" * 1000)
    (set_dir / "noise.wav").write_bytes(b"n" * 1000)
    (set_dir / "a.osu").write_text(_mania_osu("a.ogg"), encoding="utf-8")
    (set_dir / "b.osu").write_text(_mania_osu("b.mp3"), encoding="utf-8")
    prune(set_dir)
    assert layout_ok(set_dir)
    assert audio_filename(set_dir / "a.osu") == "a.ogg"
    assert audio_filename(set_dir / "b.osu") == "b.mp3"
    assert (set_dir / "a.ogg").exists() and (set_dir / "b.mp3").exists()
    assert not (set_dir / "noise.wav").exists()


def test_prepare_set_flattens_audio_and_rewrites_filename(tmp_path: Path) -> None:
    set_dir = tmp_path / "8"
    nested = set_dir / "audio"
    nested.mkdir(parents=True)
    (nested / "song.ogg").write_bytes(b"x" * 1000)
    (set_dir / "a.osu").write_text(_mania_osu("audio/song.ogg"), encoding="utf-8")
    prepare_set(set_dir)
    assert layout_ok(set_dir)
    assert (set_dir / "song.ogg").is_file()
    assert audio_filename(set_dir / "a.osu") == "song.ogg"
    assert not nested.exists()


def test_apply_4k_filter_keeps_only_mania(tmp_path: Path) -> None:
    set_dir = _write_set(tmp_path)
    (set_dir / "a.osu").write_text(_MANIA_4K, encoding="utf-8")
    (set_dir / "b.osu").write_text(_STD, encoding="utf-8")
    assert apply_4k_filter(set_dir)
    assert (set_dir / "a.osu").exists()
    assert not (set_dir / "b.osu").exists()

    set_dir2 = _write_set(tmp_path, sid="456")
    (set_dir2 / "only.osu").write_text(_STD, encoding="utf-8")
    assert not apply_4k_filter(set_dir2)


def test_fetch_page_raises_on_bad_status() -> None:
    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class _Session:
        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.calls: list[dict] = []

        def get(self, url: str, params: dict, timeout: int) -> _Resp:
            self.calls.append({"url": url, "params": params, "timeout": timeout})
            return _Resp(self.payload)

    ok = _Session({"status": 0, "data": [{"sid": 1}], "endid": 99})
    payload = fetch_page(ok, offset=0, limit=50)  # type: ignore[arg-type]
    assert payload["data"] == [{"sid": 1}]
    assert ok.calls[0]["params"] == {"L": 50, "O": 0, "T": 2, "C": 1}

    bad = _Session({"status": -1, "msg": "boom"})
    with pytest.raises(RuntimeError):
        fetch_page(bad, offset=0, limit=50)  # type: ignore[arg-type]


def test_info_has_mania_4k() -> None:
    def payload(*diffs: dict) -> dict:
        return {"status": 0, "data": {"bid_data": list(diffs)}}

    assert info_has_mania_4k(payload({"mode": 3, "CS": 4.0})) is True
    assert info_has_mania_4k(payload({"mode": 3, "CS": 7.0}, {"mode": 3, "CS": 4})) is True
    assert info_has_mania_4k(payload({"mode": 3, "CS": 7.0})) is False
    assert info_has_mania_4k(payload({"mode": 0, "CS": 4.0})) is False
    assert info_has_mania_4k({"status": -1, "data": {"bid_data": [{"mode": 3, "CS": 4}]}}) is None
    assert info_has_mania_4k({"status": 0, "data": {"bid_data": []}}) is None


def test_state_transitions_and_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector_state, "state_path", lambda: tmp_path / "state.json")

    st = collector_state.load_state()
    assert st == {
        "offset": 0,
        "completed": set(),
        "failed": {},
        "skipped_no_4k": set(),
        "skipped_no_mania": set(),
        "skipped_class": set(),
    }

    assert not collector_state.should_skip(1, st)
    collector_state.apply_result(1, "completed", st)
    collector_state.apply_result(2, "skipped_no_4k", st)
    collector_state.apply_result(3, "failed", st, "download failed")
    collector_state.apply_result(4, "skipped_no_mania", st)
    collector_state.apply_result(5, "skipped_class", st)
    assert collector_state.should_skip(1, st)
    assert collector_state.should_skip(2, st)
    assert not collector_state.should_skip(3, st)
    assert collector_state.should_skip(4, st)
    assert collector_state.should_skip(5, st)
    assert st["failed"] == {"3": "download failed"}

    collector_state.apply_result(3, "completed", st)
    assert "3" not in st["failed"]
    assert collector_state.should_skip(3, st)

    collector_state.persist(st, offset=42)
    loaded = collector_state.load_state()
    assert loaded["offset"] == 42
    assert loaded["completed"] == {1, 3}
    assert loaded["skipped_no_4k"] == {2}
    assert loaded["skipped_no_mania"] == {4}
    assert loaded["skipped_class"] == {5}
    on_disk = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert on_disk["completed"] == [1, 3]
    assert not (tmp_path / "state.json.tmp").exists()


def test_corrupt_state_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(collector_state, "state_path", lambda: path)
    with pytest.raises(SystemExit, match="corrupt collector state"):
        collector_state.load_state()


def test_safe_extract_rejects_path_escape(tmp_path: Path) -> None:
    osz = tmp_path / "bad.osz"
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(osz, "w") as zf:
        zf.writestr("../escape.txt", "nope")
    with pytest.raises(RuntimeError, match="unsafe zip path"):
        _safe_extract(osz, dest)
    assert not (tmp_path / "escape.txt").exists()


def test_run_resumes_offset_and_records_api_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.collect.sayobot as sayobot

    raw = tmp_path / "raw"
    tmp = tmp_path / "tmp"
    state_file = tmp_path / "state.json"
    seen_offsets: list[int] = []
    handle_calls: list[int] = []

    def fake_fetch(session: object, offset: int, limit: int) -> dict:
        seen_offsets.append(offset)
        return {
            "status": 0,
            "endid": 0,
            "data": [
                {"sid": 1, "modes": 8, "approved": 1},
                {"sid": 2, "modes": 1, "approved": 1},
            ],
        }

    def fake_handle(sid: int, pbar: object) -> tuple[str, None]:
        handle_calls.append(sid)
        return "completed", None

    monkeypatch.setattr(sayobot, "fetch_page", fake_fetch)
    monkeypatch.setattr(sayobot, "handle_sid", fake_handle)
    monkeypatch.setattr(sayobot, "setup_logging", lambda **k: None)
    monkeypatch.setattr(sayobot, "raw_dir", lambda: raw)
    monkeypatch.setattr(sayobot, "_tmp_dir", lambda: tmp)
    monkeypatch.setattr(collector_state, "state_path", lambda: state_file)
    monkeypatch.setattr(collector_state, "raw_dir", lambda: raw)
    collector_state.save_state(
        {
            "offset": 77,
            "completed": [1],
            "failed": {},
            "skipped_no_4k": [],
            "skipped_no_mania": [],
            "skipped_class": [],
        }
    )

    run(target=1, list_delay=0)

    assert seen_offsets == [77]
    assert handle_calls == []
    loaded = collector_state.load_state()
    assert loaded["completed"] == {1}
    assert loaded["skipped_no_mania"] == {2}


def test_run_skips_mini_when_info_has_no_4k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.collect.sayobot as sayobot

    raw = tmp_path / "raw"
    tmp = tmp_path / "tmp"
    state_file = tmp_path / "state.json"
    handle_calls: list[int] = []
    info_calls: list[int] = []

    def fake_fetch(session: object, offset: int, limit: int) -> dict:
        return {
            "status": 0,
            "endid": 0,
            "data": [
                {"sid": 10, "modes": 8, "approved": 1},
                {"sid": 11, "modes": 8, "approved": 1},
            ],
        }

    def fake_info(session: object, sid: int) -> bool | None:
        info_calls.append(sid)
        return {10: False, 11: True}[sid]

    def fake_handle(sid: int, pbar: object) -> tuple[str, None]:
        handle_calls.append(sid)
        return "completed", None

    monkeypatch.setattr(sayobot, "fetch_page", fake_fetch)
    monkeypatch.setattr(sayobot, "sid_has_mania_4k", fake_info)
    monkeypatch.setattr(sayobot, "handle_sid", fake_handle)
    monkeypatch.setattr(sayobot, "setup_logging", lambda **k: None)
    monkeypatch.setattr(sayobot, "raw_dir", lambda: raw)
    monkeypatch.setattr(sayobot, "_tmp_dir", lambda: tmp)
    monkeypatch.setattr(collector_state, "state_path", lambda: state_file)
    monkeypatch.setattr(collector_state, "raw_dir", lambda: raw)

    run(target=2, list_delay=0)

    assert info_calls == [10, 11]
    assert handle_calls == [11]
    loaded = collector_state.load_state()
    assert loaded["skipped_no_4k"] == {10}
    assert loaded["completed"] == {11}


def test_user_agent_contact_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import scripts.collect.sayobot as sayobot

    monkeypatch.setenv("AUDIO2MAP_CONTACT", "me@example.com")
    monkeypatch.delenv("SAYOBOT_USER_AGENT", raising=False)
    reloaded = importlib.reload(sayobot)
    try:
        assert "me@example.com" in reloaded.USER_AGENT
    finally:
        monkeypatch.delenv("AUDIO2MAP_CONTACT", raising=False)
        importlib.reload(sayobot)
