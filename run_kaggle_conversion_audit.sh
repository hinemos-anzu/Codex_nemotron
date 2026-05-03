#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1}"
ADAPTER_PATH="${ADAPTER_PATH:-/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20}"
BASELINE_SUBMISSION_PATH="${BASELINE_SUBMISSION_PATH:-/kaggle/working/submission.zip}"
OUTDIR="${OUTDIR:-/kaggle/working/reports/conversion_audit}"

python conversion_audit.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --baseline-submission-path "$BASELINE_SUBMISSION_PATH" \
  --outdir "$OUTDIR"

printf "\n[done] reports generated under: %s\n" "$OUTDIR"
