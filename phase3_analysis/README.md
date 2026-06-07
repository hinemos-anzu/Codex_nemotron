# Phase3 Analysis

This directory contains the Phase3 analysis artifact set for the Golden Baseline failure-category workflow.

## Safety status

- Training/SFT: **not executed**.
- Golden adapter mutation: **not executed**.
- `adapter_model.safetensors` generation/overwrite: **not executed**.
- `adapter_config.json` mutation: **not executed**.
- `submission.zip` creation: **not executed**.
- Kaggle submission/Public LB check: **not executed**.

## Inputs observed by this run

- Golden notebook/script: `b3-nemotron-svd-26042701.ipynb`
- Adapter path: `NOT_PROVIDED`
- Validation path: `NOT_PROVIDED`
- Prediction log path: `NOT_PROVIDED`
- Logprob path: `NOT_PROVIDED`

## Offline one-file notebook

- `phase3_offline_analysis.ipynb` is the self-contained Kaggle Internet-OFF notebook version for RTX Pro5000 environments.
- It uses Python standard library only and embeds the Phase3 analysis code in one code cell; it does not require importing `phase3_make_recommendation.py`.
- The notebook is analysis-only: no training, no adapter mutation, no `submission.zip` creation, and no Kaggle submission.

## Golden Baseline input/log audit

See `phase3_analysis/golden_baseline_input_audit.md` for the attached notebook inspection.  The notebook exposes the Golden adapter path (`/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`) and adapter diagnostics, but it does not define a validation/held-out dataset path and does not create a row-level Golden prediction log.  If no external per-problem Golden validation prediction artifact exists, do not modify Golden Baseline code by default; run category-map/CV-design analysis first and keep measured prediction/logprob sections as `NOT_AVAILABLE`. A logging observer is a later HOLD option only if needed, and it must not change prompt, decoding, adapter, model, or submission behavior.

## Validation source clarification

- Kaggle `train.csv` is the labeled source from which a Phase3 validation/held-out set can be made.
- Prefer a deterministic held-out split derived from `train.csv`, for example `/kaggle/working/phase3_validation_split.csv`, rather than treating the whole training file as the validation set.
- The `--validation` file must contain gold answers; Kaggle `test.csv` is not sufficient for local accuracy/failure analysis.
- Kaggle Discussion search did not yield a verified public CV split in this environment; if none is confirmed, use a deterministic split from competition `train.csv` as the first option.

## Deterministic split creation

Create the first-pass Phase3 validation file from Kaggle `train.csv` without touching Golden Baseline inference:

```bash
python phase3_make_recommendation.py \
  --output-dir phase3_analysis \
  --create-validation-from-train /kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv \
  --validation-output /kaggle/working/phase3_validation_split.csv \
  --validation-fraction 0.2 \
  --split-seed 42
```

The script preserves all original columns and writes `/kaggle/working/phase3_validation_split.manifest.json` with seed, method, and row counts. See `phase3_analysis/validation_split_instructions.md` for exact Kaggle notebook settings and the SHA-256 split algorithm.

## Execution handoff

See `phase3_analysis/execution_handoff_plan.md` for the split between repository-side assistant work and Kaggle-side user execution. The first-pass recommendation is to run category-map/CV-design mode on a deterministic `train.csv`-derived split without modifying Golden Baseline logging.

## Kaggle package review

See `phase3_analysis/kaggle_package_review.md`, `phase3_analysis/kaggle_package_2_review.md`, and `phase3_analysis/kaggle_package_3_review.md` for reviews of the uploaded Kaggle packages, including split validation, first-pass measurement status, and classifier fixes for caret exponent / Roman numeral / symbolic-operator cases.

## Troubleshooting

See `phase3_analysis/troubleshooting.md` if the Kaggle notebook raises `FileNotFoundError`, reports `validation_rows: 0`, creates an empty category map, or reports `NOT_AVAILABLE` metrics in the first pass.

## Classification rules

Categories are assigned by deterministic keyword/rule matching.  Cryptarithm, bit manipulation, and numeral conversion rules are evaluated before broader categories.
Ambiguous rows are marked `manual_review_required=True` instead of being silently treated as reliable `other` examples.

### High-priority category rules

- `cryptarithm`: alphametic/cryptarithm keywords, distinct digit/letter mapping language, or uppercase letter equations such as `SEND + MORE = MONEY`.
- `bit_manipulation`: bitwise operator keywords and symbols (`xor`, `&`, `|`, `^`, `<<`, `>>`, mask, signed/unsigned).
- `numeral_conversion`: explicit base/radix conversion, binary/hex literals, decimal/hex/binary/Roman conversion language.

## Measurement status

- Validation rows analyzed: `0`
- Golden validation accuracy: `NOT_AVAILABLE`
- Parse success rate: `NOT_AVAILABLE`
- Status: `NOT_MEASURED_NO_VALIDATION_OR_PREDICTION_LOGS`
