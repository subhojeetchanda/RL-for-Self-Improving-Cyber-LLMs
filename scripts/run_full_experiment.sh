#!/usr/bin/env bash
# Reproduce one paired SFT / MAPPO experiment on the frozen exploratory splits.
# Usage: bash scripts/run_full_experiment.sh SEED [EPISODES]
set -euo pipefail

SEED="${1:?usage: run_full_experiment.sh SEED [EPISODES]}"
EPISODES="${2:-256}"
cd "$(dirname "$0")/.."

TRAIN=data/sft/train_experiment.jsonl
VAL=data/sft/validation_experiment.jsonl
BENCH=data/evaluation/four_quadrants_confirmatory.jsonl

python3 scripts/build_confirmatory_splits.py --seed 42 >/dev/null  # regenerates identical frozen files

SFT_DIR="checkpoints/sft_seed${SEED}_strong"
MAPPO_DIR="checkpoints/mappo_seed${SEED}_strong"

python3 src/train_sft.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --train-file "$TRAIN" --validation-file "$VAL" \
  --output-dir "$SFT_DIR" \
  --epochs 10 --batch-size 2 --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 --max-length 512 --max-new-tokens 64 \
  --validation-samples 2 --seed "$SEED"

python3 scripts/run_llm_mappo.py \
  --base-model Qwen/Qwen2.5-0.5B \
  --sft-adapter "$SFT_DIR" \
  --train-file "$TRAIN" --validation-file "$VAL" \
  --output-dir "$MAPPO_DIR" \
  --episodes "$EPISODES" --update-every 16 --ppo-epochs 2 \
  --max-new-tokens 64 --validation-samples 2 --evaluation-interval 128 \
  --seed "$SEED"

: "${EVAL_CANARY_VALUE:?export EVAL_CANARY_VALUE before evaluation (canary is never stored in the repo)}"
python3 scripts/evaluate_llm_four_quadrants.py \
  --benchmark "$BENCH" --train-file "$TRAIN" --validation-file "$VAL" \
  --sft-adapter "$SFT_DIR" \
  --mappo-adapter "$MAPPO_DIR" \
  --benchmark-status confirmatory --samples-per-case 5 \
  --max-new-tokens 64 --seed "$SEED" \
  --output "reports/llm_eval_seed${SEED}.json"

python3 scripts/analyze_llm_four_quadrant_report.py \
  "reports/llm_eval_seed${SEED}.json" \
  --output "reports/full_metrics_seed${SEED}.json" \
  --markdown "reports/full_metrics_seed${SEED}.md"
