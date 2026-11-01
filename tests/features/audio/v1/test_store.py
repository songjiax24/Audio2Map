"""Rate-1 PCM and mel manifest roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.features.audio.v1.constants import N_MELS
from audio2map.features.audio.v1.store import (
    _append_record,
    _complete,
    _dump,
    _ensure_header,
    _mel_rel,
    _pcm_rel,
    load_audio_cache,
)


def test_pcm_and_mel_roundtrip(tmp_path: Path) -> None:
    pcm = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    n_cols = 2 * (1 - (-1)) + 3
    mel = np.arange(N_MELS * n_cols, dtype=np.float32).reshape(N_MELS, n_cols)
    _ensure_header(tmp_path)
    pcm_file = _pcm_rel("set/a.mp3")
    mel_file = _mel_rel("set/a.mp3", -1, 1)
    _dump(tmp_path / pcm_file, pcm)
    _dump(tmp_path / mel_file, mel)
    _append_record(
        tmp_path,
        {
            "audio": "set/a.mp3",
            "t_audio_ms": 1000.0,
            "source_bytes": 50,
            "n_pcm": 16,
            "pcm": pcm_file,
            "mels": [{"frame_lo": -1, "frame_hi": 1, "file": mel_file}],
        },
    )
    _append_record(tmp_path, {"audio": "set/b.mp3", "error": "RuntimeError: boom"})
    cache = load_audio_cache(tmp_path)
    loaded, duration = cache.pcm("set/a.mp3", 50)
    assert duration == 1000.0
    assert np.array_equal(loaded, pcm)
    assert cache.pcm("set/a.mp3", 49) is None
    assert cache.pcm("set/b.mp3", 50) is None
    got = cache.mel("set/a.mp3", 1000.0, -1, 1)
    assert got is not None
    assert np.array_equal(got, mel)
    assert cache.mel("set/a.mp3", 1002.0, -1, 1) is None
    source = tmp_path / "src.bin"
    source.write_bytes(b"x" * 50)
    assert _complete(cache, tmp_path, source, "set/a.mp3")
    assert not _complete(cache, tmp_path, source, "set/b.mp3")
