#!/usr/bin/env bash
# Full formal pipeline: chart meta → v3 audio grid → audit → train.
set -euo pipefail

ROOT="/root/audio2map"
DATA="/root/autodl-tmp/audio2map_data"
LOG_DIR="$DATA/processed_v2/logs"
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
echo "disk: $(df -h /root/autodl-tmp | tail -1)"
echo "============================================================"

echo
echo "[1/4] compute chart meta manifest (eligible charts, fresh write)"
echo "  log: $META_LOG"
.venv/bin/python -u scripts/compute_chart_meta.py --eligible-only 2>&1 | tee "$META_LOG"
META_LINES="$(wc -l < "$DATA/chart_meta/manifest.jsonl")"
echo "  manifest lines: $META_LINES"

echo
echo "[2/4] precompute audio grid v3 (128-d, --no-skip-existing)"
echo "  log: $GRID_LOG"
.venv/bin/python -u scripts/precompute_audio_grid.py --no-skip-existing 2>&1 | tee "$GRID_LOG"

echo
echo "[3/4] audit audio grid"
.venv/bin/python scripts/audit_audio_grid.py --fix || true
.venv/bin/python scripts/audit_audio_grid.py

echo
echo "[4/4] formal training (200k steps, batch=16)"
echo "  log: $TRAIN_LOG"
.venv/bin/python -u scripts/train_v2.py \
  --d-model 512 \
  --n-heads 8 \
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
echo "checkpoints: $DATA/processed_v2/checkpoints/formal_enc_dec_v3/"
echo "============================================================"
