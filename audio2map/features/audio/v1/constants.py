"""Shared v1 audio constants. Intervals are derived from the rate and hop."""

from __future__ import annotations

SAMPLE_RATE = 22_050
HOP = 128
N_FFT = 1024
N_MELS = 128
FMIN_HZ = 20.0
# Two hops, in milliseconds: 256 / 22050 * 1000.
DT_MS = (2 * HOP) / SAMPLE_RATE * 1000.0
# Snap ceil(t / Δt) when the quotient is already an integer within this tolerance.
# This is a mel-frame index, not a beat-coverage time.
FRAME_INDEX_SNAP = 1e-8
MEL_FLOOR = 1e-5
MEL_NORM_VERSION = "MEL_NORM_V1"
# tools/compute_mel_norm_v1.py on 2026-10-22.
# beatmapset_v1 train, seed 0, ratios 0.8/0.1/0.1, one audio file per group,
# numpy Generator(0) shuffle, first 50. Population mean and std (ddof=0) of
# ln(mel + 1e-5) over every mel bin and every frame of the v1 buffer for
# [0, T_audio). ffmpeg 7.1.1, soxr 1.1.0. n_values = 216116992.
MEL_NORM_MU = -2.0295708833
MEL_NORM_SIGMA = 3.2925165658
