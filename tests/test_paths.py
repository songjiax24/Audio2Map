"""Path helper behaviour (no data disk required)."""

from __future__ import annotations

import logging
from pathlib import Path

from audio2map.utils import paths as paths_mod


def test_get_data_root_respects_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", str(tmp_path))
    assert paths_mod.get_data_root() == tmp_path.resolve()
    assert paths_mod.raw_dir() == tmp_path.resolve() / "raw"
    assert paths_mod.formal_checkpoint_dir() == (
        tmp_path.resolve() / "processed" / "checkpoints" / paths_mod.FORMAL_CHECKPOINT_NAME
    )


def test_get_data_root_expands_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", "~/audio2map_data")
    assert paths_mod.get_data_root() == (tmp_path / "audio2map_data").resolve()


def test_blank_env_falls_back_to_repo_local(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AUDIO2MAP_DATA_ROOT", "  ")
    monkeypatch.setattr(paths_mod, "_WARNED_UNSET", False)
    with caplog.at_level(logging.WARNING, logger="audio2map.utils.paths"):
        assert paths_mod.get_data_root() == paths_mod._REPO_LOCAL_ROOT


def test_unset_env_falls_back_to_repo_local(monkeypatch, caplog) -> None:
    monkeypatch.delenv("AUDIO2MAP_DATA_ROOT", raising=False)
    monkeypatch.setattr(paths_mod, "_WARNED_UNSET", False)
    with caplog.at_level(logging.WARNING, logger="audio2map.utils.paths"):
        root1 = paths_mod.get_data_root()
        root2 = paths_mod.get_data_root()
    assert root1 == root2 == paths_mod._REPO_LOCAL_ROOT
    assert sum(1 for r in caplog.records if "AUDIO2MAP_DATA_ROOT is unset" in r.message) == 1


def test_path_is_dir_treats_permission_error_as_missing(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "blocked"

    def boom(self: Path) -> bool:  # noqa: ARG001
        raise PermissionError(13, "Permission denied", str(blocked))

    monkeypatch.setattr(Path, "is_dir", boom)
    assert paths_mod._path_is_dir(blocked) is False
