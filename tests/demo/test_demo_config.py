"""Demo path/device helpers (no GPU or ffmpeg)."""

from __future__ import annotations

from pathlib import Path

from demo.backend.config import (
    FORMAL_CHECKPOINT_FILE,
    get_checkpoint_path,
    get_manifest_path,
    resolve_device_name,
)


def test_resolve_device_name_honors_cpu(monkeypatch) -> None:
    monkeypatch.setenv("AUDIO2MAP_DEVICE", "cpu")
    assert resolve_device_name() == "cpu"
    monkeypatch.setenv("AUDIO2MAP_DEVICE", "CUDA")
    assert resolve_device_name() == "cuda"


def test_checkpoint_prefers_env(monkeypatch, tmp_path: Path) -> None:
    ckpt = tmp_path / "custom.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("AUDIO2MAP_CHECKPOINT", str(ckpt))
    assert get_checkpoint_path() == ckpt.resolve()


def test_checkpoint_prefers_formal_then_latest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("AUDIO2MAP_CHECKPOINT", raising=False)
    formal = tmp_path / "processed" / "checkpoints" / "formal_enc_dec_v3"
    formal.mkdir(parents=True)
    (formal / "step_10.pt").write_bytes(b"a")
    (formal / "step_2000.pt").write_bytes(b"b")
    assert get_checkpoint_path() == (formal / "step_2000.pt").resolve()

    exact = formal / FORMAL_CHECKPOINT_FILE
    exact.write_bytes(b"c")
    assert get_checkpoint_path() == exact.resolve()


def test_checkpoint_falls_back_to_repo_copy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("AUDIO2MAP_CHECKPOINT", raising=False)
    monkeypatch.setattr("demo.backend.config.DEFAULT_CHECKPOINT", tmp_path / "repo.pt")
    (tmp_path / "repo.pt").write_bytes(b"x")
    assert get_checkpoint_path() == tmp_path / "repo.pt"


def test_manifest_prefers_data_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUDIO2MAP_CHART_META_MANIFEST", raising=False)
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", str(tmp_path))
    meta = tmp_path / "chart_meta"
    meta.mkdir()
    path = meta / "manifest.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert get_manifest_path() == path.resolve()


def test_checkpoint_missing_points_at_formal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("AUDIO2MAP_CHECKPOINT", raising=False)
    monkeypatch.setattr("demo.backend.config.DEFAULT_CHECKPOINT", tmp_path / "missing-repo.pt")
    expected = tmp_path / "processed" / "checkpoints" / "formal_enc_dec_v3" / FORMAL_CHECKPOINT_FILE
    assert get_checkpoint_path() == expected
