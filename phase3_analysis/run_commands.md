# Phase3 Run Commands

## Commands executed

```bash
python phase3_make_recommendation.py --output-dir phase3_analysis
```

Actual command line captured by script:

```bash
phase3_make_recommendation.py --output-dir phase3_analysis
```

## Inputs

- Golden notebook/script: `b3-nemotron-svd-26042701.ipynb`
- Adapter path: `NOT_PROVIDED`
- Validation path: `NOT_PROVIDED`
- Prediction path: `NOT_PROVIDED`
- Logprob path: `NOT_PROVIDED`

## Repro settings

- Seed: `NOT_PROVIDED`
- Generation config: `Golden Baseline unchanged; not executed by this script`
- Script version: `phase3_make_recommendation.py stdlib-v1`
- Git hash: `5f88ccaf9713ec0acf0f2b2fb8ff6ec83e6520af`
- Execution datetime UTC: `2026-06-06T00:24:13.299118+00:00`

## Generated artifacts

- `phase3_analysis/README.md`
- `phase3_analysis/category_failure_summary.csv`
- `phase3_analysis/category_map.csv`
- `phase3_analysis/failure_cases_bit_manipulation.csv`
- `phase3_analysis/failure_cases_cryptarithm.csv`
- `phase3_analysis/failure_cases_numeral_conversion.csv`
- `phase3_analysis/failure_type_summary.csv`
- `phase3_analysis/golden_validation_predictions.jsonl`
- `phase3_analysis/golden_validation_summary.csv`
- `phase3_analysis/min_logprob_summary.csv`
- `phase3_analysis/phase3_recommendation.md`
- `phase3_analysis/reproducibility_notes.md`
- `phase3_analysis/run_commands.md`
- `phase3_analysis/validation_set_labeled.csv`

## Notebook audit command

The attached notebook was inspected as source text only to answer where validation inputs and prediction logs are defined. No notebook cell was executed.

```bash
python - <<'PY'
import json, pathlib
src = ''.join(json.loads(pathlib.Path('b3-nemotron-svd-26042701.ipynb').read_text())['cells'][1]['source'])
for token in ['ADAPTER_PATH', 'MODEL_PATH', 'DIAG_DIR', 'SUBMISSION_ZIP', 'validation', 'golden_validation_predictions']:
    print(token, token in src)
PY
```

## Offline notebook validation command

The generated Kaggle notebook was validated structurally and compiled as Python source without executing the notebook cell.

```bash
python - <<'PY'
import json
nb = json.load(open('phase3_offline_analysis.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][0]['source'])
compile(code, 'phase3_offline_analysis.ipynb', 'exec')
print(nb['nbformat'], len(nb['cells']), nb['cells'][0]['cell_type'], len(code.splitlines()))
PY
```

## Deterministic split smoke test command

A tiny local CSV was used to verify deterministic split creation without Kaggle data or model execution.

```bash
python phase3_make_recommendation.py \
  --output-dir /tmp/phase3_split_test/out \
  --create-validation-from-train /tmp/phase3_split_test/train.csv \
  --validation-output /tmp/phase3_split_test/phase3_validation_split.csv \
  --validation-fraction 0.4 \
  --split-seed 42
```
