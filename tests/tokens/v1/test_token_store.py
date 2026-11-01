"""Rate-1 token cache round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio2map.tokens.v1.store import TokenRecord, _write_token_cache, load_token_cache
from audio2map.tokens.v1.vocab import build_vocab


def test_roundtrip_reads_the_stored_string(tmp_path: Path) -> None:
    vocab = build_vocab()
    ids = np.array(
        [vocab["<BOS>"], vocab["<BEAT>"], vocab["<EOS>"]],
        dtype=np.int16,
    )
    records = {"a.osu": TokenRecord("a.osu", 1000.0, 0, 3)}
    _write_token_cache(tmp_path, records, {"a.osu": ids})
    loaded = load_token_cache(tmp_path)
    assert loaded.tokens("a.osu", 1000.0) == ["<BOS>", "<BEAT>", "<EOS>"]
    assert loaded.tokens("a.osu", 1001.0) is None
    assert loaded.tokens("missing.osu", 1000.0) is None


def test_error_row_is_not_a_hit(tmp_path: Path) -> None:
    records = {
        "bad.osu": TokenRecord("bad.osu", 10.0, 0, 0, error="ChartRejected: no"),
    }
    _write_token_cache(tmp_path, records, {})
    loaded = load_token_cache(tmp_path)
    assert loaded.tokens("bad.osu", 10.0) is None
