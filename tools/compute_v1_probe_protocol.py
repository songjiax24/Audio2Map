#!/usr/bin/env python3
"""Choose pattern probe levels from a deterministic train-only calibration subset."""

import argparse
import logging
from pathlib import Path

from audio2map.eval.v1.protocol import CalibrationConfig, calibrate_patterns

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--training-vocab", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--charts", type=int, default=CalibrationConfig.charts)
    parser.add_argument("--seed", type=int, default=CalibrationConfig.seed)
    parser.add_argument("--workers", type=int, default=CalibrationConfig.workers)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = calibrate_patterns(CalibrationConfig(
        args.source_run, args.training_vocab, args.out, args.charts, args.seed, args.workers,
    ))
    for field, row in result["calibration"]["statistics"].items():
        logger.info("%s n=%s positive=%s levels=%s", field, row["distribution"]["count"],
                    row["positive_count"], row["levels"])


if __name__ == "__main__":
    main()
