#!/usr/bin/env bash
set -Eeuo pipefail

# Kaggle-safe runner for conversion_audit.py
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1}"
ADAPTER_PATH="${ADAPTER_PATH:-/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20}"
BASELINE_SUBMISSION_PATH="${BASELINE_SUBMISSION_PATH:-/kaggle/working/submission.zip}"
OUTDIR="${OUTDIR:-/kaggle/working/reports/conversion_audit}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-${SCRIPT_DIR}/conversion_audit.py}"
REPORTS_ZIP_PATH="${REPORTS_ZIP_PATH:-/kaggle/working/conversion_audit_reports.zip}"

if [[ ! -f "${AUDIT_SCRIPT}" ]]; then
  echo "[error] conversion_audit.py not found: ${AUDIT_SCRIPT}" >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] python runtime not found: ${PYTHON_BIN}" >&2
  exit 1
fi

# Ensure runtime dependency exists in Kaggle sessions.
if ! "${PYTHON_BIN}" -c "import safetensors" >/dev/null 2>&1; then
  echo "[info] installing missing dependency: safetensors"
  "${PYTHON_BIN}" -m pip install -q safetensors
fi

mkdir -p "${OUTDIR}"

"${PYTHON_BIN}" "${AUDIT_SCRIPT}"   --model-path "${MODEL_PATH}"   --adapter-path "${ADAPTER_PATH}"   --baseline-submission-path "${BASELINE_SUBMISSION_PATH}"   --outdir "${OUTDIR}" \
  --reports-zip-path "${REPORTS_ZIP_PATH}"

printf "\n[done] reports generated under: %s\n" "${OUTDIR}"
printf "[done] reports zip: %s\n" "${REPORTS_ZIP_PATH}"
