"""Training / model constants shared without importing ``torch`` modules."""

from __future__ import annotations

from audio2map.audio.tick_features import AUDIO_FEATURE_SPEC_VERSION
from audio2map.data.cond_vec import COND_VEC_DIM, COND_VEC_VERSION
from audio2map.osu.grid_config import TICKS_PER_BAR, WINDOW_BARS
from audio2map.osu.row_tokens import TOKENIZER_VERSION

# Formal default — set from ``scripts/analyze_window_token_len.py`` statistics.
MAX_DECODER_LEN = 2048
MAX_AUDIO_TICKS = WINDOW_BARS * TICKS_PER_BAR

# Report overflow against these limits in dataset stats / analysis scripts.
DECODER_LEN_REPORT_LIMITS: tuple[int, ...] = (1024, 2048, 4096)


def model_data_versions() -> dict[str, int | str]:
    """Version tags persisted in checkpoints for reproducibility."""
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "cond_vec_version": COND_VEC_VERSION,
        "cond_vec_dim": COND_VEC_DIM,
        "audio_feature_spec_version": AUDIO_FEATURE_SPEC_VERSION,
        "ticks_per_bar": TICKS_PER_BAR,
        "window_bars": WINDOW_BARS,
    }
