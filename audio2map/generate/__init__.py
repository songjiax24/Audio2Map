"""v0 chart generation: 16-bar overlap, cond selection, ``generate_chart``.

v1 sampling lives in ``audio2map.generate.v1`` and does not use this stitch.
``teacher_decode`` is the gold-prefix readout, not this package.
"""

from audio2map.generate.overlap import DecodeConfig, OverlapConfig
from audio2map.generate.service import generate_chart

__all__ = [
    "DecodeConfig",
    "OverlapConfig",
    "generate_chart",
]
