"""v1 decode, mix, and mel-frame index. Stem equality lives in tests/model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio2map.features.audio.v1.constants import DT_MS, FRAME_INDEX_SNAP
from audio2map.features.audio.v1.decode import (
    mix_equal,
    resample_hq,
    stream_from_probe,
    timeline_broken,
)
from audio2map.features.audio.v1.mel import frame_index


def test_timeline_broken_matches_ffmpeg_7_1_1() -> None:
    packet = "Error submitting packet to decoder: Invalid data found when processing input"
    assert timeline_broken(packet)
    assert timeline_broken("Header missing")
    assert timeline_broken("Header missing\n" + packet)
    bounce = "Application provided invalid, non monotonically increasing dts to muxer in stream 0"
    assert not timeline_broken(bounce)
    assert not timeline_broken("timestamp discontinuity of 448 samples")


def test_equal_weight_mix_and_resample_length() -> None:
    pcm = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 5.0], dtype=np.float32)
    mono = mix_equal(pcm, 2)
    np.testing.assert_allclose(mono, [2.0, 3.0, 4.0])
    y = resample_hq(mono, 44_100, 22_050)
    assert y.shape[0] == int(round(3 * 22_050 / 44_100))
    same = resample_hq(mono, 22_050, 22_050)
    np.testing.assert_array_equal(same, mono)


def test_stream_from_probe_reads_rate_and_channels() -> None:
    stereo = {"streams": [{"sample_rate": "44100", "channels": 2}]}
    assert stream_from_probe(stereo) == (44100, 2)
    surround = {"streams": [{"sample_rate": "48000", "channels": 6}]}
    assert stream_from_probe(surround) == (48000, 6)


def test_frame_index_snaps_binary_error() -> None:
    on_boundary = 3 * DT_MS
    assert frame_index(on_boundary) == 3
    nudged = on_boundary + DT_MS * (FRAME_INDEX_SNAP / 2)
    assert frame_index(nudged) == 3
    past = on_boundary + DT_MS * (FRAME_INDEX_SNAP * 10)
    assert frame_index(past) == 4


@pytest.mark.requires_data
def test_known_files_match_the_verified_timeline_logs(data_raw_dir: Path) -> None:
    from audio2map.features.audio.v1.decode import decode_audio

    broken = ("2322253/audio.ogg", "1743718/audio.ogg", "2298998/audio.mp3")
    for relative in broken:
        decoded = decode_audio(data_raw_dir / relative)
        assert decoded.timeline_broken, relative
    bounce = decode_audio(data_raw_dir / "2080086/audio.ogg")
    assert not bounce.timeline_broken
    assert bounce.t_audio_ms == float(bounce.y.shape[0]) / bounce.sample_rate * 1000.0
