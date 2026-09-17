"""Demo upload keeps the original audio file (no ffmpeg / mp3 transcode)."""

from __future__ import annotations

from pathlib import Path

import pytest

from demo.backend.inference_service import InferenceService


def test_save_upload_keeps_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("demo.backend.inference_service.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("demo.backend.inference_service.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("demo.backend.inference_service.audio_grid_dir", lambda: tmp_path / "grid")
    svc = InferenceService()
    file_id = svc.save_upload("song.WAV", b"RIFF-fake")
    stored = svc._upload_path(file_id)
    assert stored.name == "song.wav"
    assert stored.read_bytes() == b"RIFF-fake"

    file_id2 = svc.save_upload("track.mp3", b"ID3-fake")
    stored2 = svc._upload_path(file_id2)
    assert stored2.name == "track.mp3"
    assert stored2.read_bytes() == b"ID3-fake"


def test_save_upload_rejects_ogg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("demo.backend.inference_service.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("demo.backend.inference_service.OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr("demo.backend.inference_service.audio_grid_dir", lambda: tmp_path / "grid")
    svc = InferenceService()
    with pytest.raises(ValueError, match="Unsupported"):
        svc.save_upload("a.ogg", b"x")


def test_download_path_uses_export_files(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr("demo.backend.inference_service.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("demo.backend.inference_service.OUTPUT_DIR", out)
    monkeypatch.setattr("demo.backend.inference_service.audio_grid_dir", lambda: tmp_path / "grid")
    svc = InferenceService()
    job = out / "abc"
    job.mkdir(parents=True)
    osu = job / "Artist - Title (Audio2Map) [Generated].osu"
    osz = job / "Artist - Title.osz"
    osu.write_text("x", encoding="utf-8")
    osz.write_bytes(b"PK")
    (job / "export_files.json").write_text(
        '{"osu": "Artist - Title (Audio2Map) [Generated].osu", "osz": "Artist - Title.osz"}',
        encoding="utf-8",
    )
    assert svc.download_path("abc", "osu") == osu
    assert svc.download_path("abc", "osz") == osz


def test_job_status_includes_progress(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr("demo.backend.inference_service.UPLOAD_DIR", tmp_path)
    monkeypatch.setattr("demo.backend.inference_service.OUTPUT_DIR", out)
    monkeypatch.setattr("demo.backend.inference_service.audio_grid_dir", lambda: tmp_path / "grid")
    svc = InferenceService()
    job = out / "abc"
    job.mkdir(parents=True)
    from demo.backend.inference_service import JobState

    svc._jobs["abc"] = JobState(
        job_id="abc",
        status="running",
        step="inference",
        windows_done=3,
        windows_total=12,
    )
    payload = svc.job_status("abc")
    assert payload["windows_done"] == 3
    assert payload["windows_total"] == 12
    assert payload["status"] == "running"
