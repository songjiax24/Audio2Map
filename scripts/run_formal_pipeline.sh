#!/usr/bin/env bash
# Full formal pipeline: chart meta → v3 audio grid → audit → train.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${AUDIO2MAP_DATA_ROOT:-}" ]]; then
  echo "Set AUDIO2MAP_DATA_ROOT to your data disk (raw/, processed/, chart_meta/)." >&2
  exit 1
fi
DATA="$AUDIO2MAP_DATA_ROOT"
LOG_DIR="$DATA/processed/logs"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOG_DIR/formal_pipeline_${TS}.log"
META_LOG="$LOG_DIR/compute_chart_meta_${TS}.log"
GRID_LOG="$LOG_DIR/precompute_audio_grid_v3_${TS}.log"
TRAIN_LOG="$LOG_DIR/train_v2_formal_${TS}.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

exec > >(tee -a "$PIPE_LOG") 2>&1

echo "============================================================"
echo "audio2map formal pipeline started at $(date -Is)"
echo "master log: $PIPE_LOG"
echo "disk: $(df -h "$DATA" 2>/dev/null | tail -1)"
echo "============================================================"

echo
echo "[1/4] compute chart meta manifest (eligible charts, fresh write)"
echo "  log: $META_LOG"
.venv/bin/python -u -m audio2map.cli.meta --eligible-only 2>&1 | tee "$META_LOG"
META_LINES="$(wc -l < "$DATA/chart_meta/manifest.jsonl")"
echo "  manifest lines: $META_LINES"

echo
echo "[2/4] precompute audio grid v3 (128-d, --no-skip-existing)"
echo "  log: $GRID_LOG"
.venv/bin/python -u -m audio2map.cli.grid precompute --no-skip-existing 2>&1 | tee "$GRID_LOG"

echo
echo "[3/4] audit audio grid"
.venv/bin/python -m audio2map.cli.grid audit --fix || true
.venv/bin/python -m audio2map.cli.grid audit

echo
echo "[4/4] formal training (200k steps, batch=16)"
echo "  log: $TRAIN_LOG"
.venv/bin/python -u -m audio2map.cli.train \
  --d-model 512 \
  --heads 8 \
  --encoder-layers 4 \
  --decoder-layers 6 \
  --max-decoder-len 2048 \
  --audio-dim 128 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --dropout 0.1 \
  --warmup-steps 2000 \
  --max-steps 200000 \
  --precision bf16 \
  --samples-per-chart 4 \
  --save-every 2000 \
  --log-every 50 \
  --num-workers 8 \
  2>&1 | tee "$TRAIN_LOG"

echo
echo "============================================================"
echo "pipeline finished at $(date -Is)"
echo "checkpoints: $DATA/processed/checkpoints/formal_enc_dec_v3/"
echo "============================================================"
