"""ffmpeg decode at the native rate, then soxr HQ to the model rate.

``T_audio`` is ``len(y) / sample_rate * 1000`` from this decode. Callers do
not read a second duration from ``soundfile``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio2map.features.audio.v1.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Decoder failures on the excluded sets (ffmpeg 7.1.1 pcm decode).
# Bounce logs only "non monotonically increasing dts" and must stay.
_PACKET_FAILURES = (
    "invalid data found when processing input",
    "error submitting packet to decoder",
    "header missing",
)


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """One ffmpeg decode plus the soxr resample used for the model rate."""

    y: np.ndarray
    sample_rate: int
    native_rate: int
    n_native: int
    ffmpeg_version: str
    soxr_version: str
    timeline_broken: bool
    ffmpeg_log: str

    @property
    def t_audio_ms(self) -> float:
        """Duration of this PCM buffer. Same decode that feeds the mel."""
        return float(self.y.shape[0]) / float(self.sample_rate) * 1000.0


def ffmpeg_binary() -> str:
    env = os.environ.get("AUDIO2MAP_FFMPEG", "").strip()
    if env:
        return env
    pinned = Path.home() / ".local" / "ffmpeg-pin" / "bin" / "ffmpeg"
    if pinned.is_file():
        return str(pinned)
    return "ffmpeg"


def ffprobe_binary(ffmpeg: str | None = None) -> str:
    sibling = Path(ffmpeg or ffmpeg_binary()).with_name("ffprobe")
    if sibling.is_file():
        return str(sibling)
    return "ffprobe"


def ffmpeg_version_line(binary: str | None = None) -> str:
    proc = subprocess.run(
        [binary or ffmpeg_binary(), "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    line = (proc.stdout or proc.stderr or "").splitlines()
    return line[0] if line else ""


def stream_from_probe(payload: dict) -> tuple[int, int]:
    """``(sample_rate, channels)`` from an ffprobe JSON document."""
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError("ffprobe returned no audio stream")
    stream = streams[0]
    rate = int(stream["sample_rate"])
    channels = int(stream["channels"])
    if rate <= 0 or channels <= 0:
        raise ValueError(f"ffprobe stream rate={rate} channels={channels}")
    return rate, channels


def probe_audio(path: Path, *, ffprobe: str) -> tuple[int, int]:
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr[-400:]}")
    return stream_from_probe(json.loads(proc.stdout or "{}"))


def mix_equal(pcm: np.ndarray, channels: int) -> np.ndarray:
    """Equal-weight mean across channels. ``pcm`` is interleaved float32."""
    if channels < 1:
        raise ValueError("channels must be positive")
    usable = (int(pcm.shape[0]) // channels) * channels
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    shaped = np.asarray(pcm[:usable], dtype=np.float32).reshape(-1, channels)
    if channels == 1:
        return np.ascontiguousarray(shaped[:, 0])
    return np.ascontiguousarray(shaped.mean(axis=1, dtype=np.float64).astype(np.float32))


def resample_hq(mono: np.ndarray, native_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """soxr HQ resample. Length is ``round(n_native * target_rate / native_rate)``."""
    n_native = int(mono.shape[0])
    target_n = int(round(n_native * target_rate / native_rate))
    if n_native == 0 or target_n == 0:
        return np.zeros(target_n, dtype=np.float32)
    if native_rate == target_rate:
        y = np.asarray(mono, dtype=np.float32)
    else:
        import soxr

        y = np.asarray(soxr.resample(np.asarray(mono, dtype=np.float32), native_rate, target_rate))
        y = np.asarray(y, dtype=np.float32)
    if y.shape[0] > target_n:
        y = y[:target_n]
    elif y.shape[0] < target_n:
        y = np.pad(y, (0, target_n - int(y.shape[0])))
    return np.ascontiguousarray(y, dtype=np.float32)


def timeline_broken(ffmpeg_log: str) -> bool:
    """True when ffmpeg 7.1.1 reports a decoder packet failure.

    The excluded sets log ``Invalid data found when processing input``
    (and, for the mp3, ``Header missing``). They do not log a timestamp
    discontinuity. Bounce logs only ``non monotonically increasing dts``
    and stays in the corpus, so that line alone is not a failure.
    """
    lowered = ffmpeg_log.lower()
    return any(phrase in lowered for phrase in _PACKET_FAILURES)


def decode_audio(path: Path, *, binary: str | None = None) -> DecodedAudio:
    """Decode with ffmpeg at the native rate, then soxr-resample to 22050 Hz."""
    import soxr

    exe = binary or ffmpeg_binary()
    source = Path(path)
    native_rate, channels = probe_audio(source, ffprobe=ffprobe_binary(exe))
    decoded = subprocess.run(
        [
            exe,
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(source),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    log = decoded.stderr.decode("utf-8", "replace")
    if decoded.returncode != 0 and not decoded.stdout:
        raise RuntimeError(f"ffmpeg failed to decode {source}: {log[-400:]}")
    broken = timeline_broken(log)
    if broken:
        logger.warning("timeline broken: %s", source)
    pcm = np.frombuffer(decoded.stdout, dtype="<f4")
    mono = mix_equal(pcm, channels)
    y = resample_hq(mono, native_rate)
    return DecodedAudio(
        y=y,
        sample_rate=SAMPLE_RATE,
        native_rate=native_rate,
        n_native=int(mono.shape[0]),
        ffmpeg_version=ffmpeg_version_line(exe),
        soxr_version=str(soxr.__version__),
        timeline_broken=broken,
        ffmpeg_log=log,
    )
