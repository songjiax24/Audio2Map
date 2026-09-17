"""Model / data contract constants shared without importing ``torch`` modules.

``WINDOW_BARS`` is the train/infer slice, not a property of the tick lattice.
"""

from __future__ import annotations

from audio2map.features.audio.tick_features import AUDIO_FEATURE_SPEC_VERSION
from audio2map.features.cond import COND_VEC_DIM, COND_VEC_VERSION
from audio2map.grid import TICKS_PER_BAR
from audio2map.tokens import TOKENIZER_VERSION

WINDOW_BARS = 16
# Formal default — set from ``tools/analyze_window_token_len.py`` statistics.
MAX_DECODER_LEN = 2048
MAX_AUDIO_TICKS = WINDOW_BARS * TICKS_PER_BAR


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
