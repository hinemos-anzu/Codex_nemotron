# Deterministic Validation Split Instructions

This file explains exactly how to create `phase3_validation_split.csv` from the competition `train.csv` for the first Phase3 pass.

## Goal

Create a fixed, reproducible, gold-labeled validation subset without touching Golden Baseline inference or adapters.

- Source: `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv`
- Derived validation file: `/kaggle/working/phase3_validation_split.csv`
- Default split: 20% held-out by deterministic SHA-256 hash
- Default seed: `42`

## Recommended Kaggle notebook setting

In `phase3_offline_analysis.ipynb`, set only this part of `NOTEBOOK_CONFIG`:

```python
NOTEBOOK_CONFIG = {
    "output_dir": "phase3_analysis",
    "create_validation_split": True,
    "train_source": "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv",
    "validation_output": "/kaggle/working/phase3_validation_split.csv",
    "validation_fraction": 0.2,
    "validation_size": None,
    "split_seed": "42",
    "validation": "",
    "predictions": "",
    "logprobs": "",
    "adapter_path": "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
}
```

When `create_validation_split=True` and `validation=""`, the notebook creates the split first, then uses that file as the Phase3 validation input.

## Equivalent CLI command

If running the script instead of the notebook:

```bash
python phase3_make_recommendation.py \
  --output-dir phase3_analysis \
  --create-validation-from-train /kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv \
  --validation-output /kaggle/working/phase3_validation_split.csv \
  --validation-fraction 0.2 \
  --split-seed 42
```

## Exact split algorithm

1. Read `train.csv` with `csv.DictReader`.
2. For each row, choose a stable row key in this order:
   - `problem_id`
   - `id`
   - `question_id`
   - `row_id`
   - `uuid`
   - detected question/prompt/problem text
   - full JSON row as a fallback
3. Compute `sha256(seed + "\\0" + stable_row_key)`.
4. Sort rows by `(hash, original_index)`.
5. Take the first `round(n_total * validation_fraction)` rows as validation.
6. Preserve all original columns exactly as they appeared in `train.csv`.
7. Write a manifest next to the split file:
   - `/kaggle/working/phase3_validation_split.manifest.json`

## What the manifest records

The manifest records:

- source train path
- validation output path
- seed
- fraction or exact size
- total rows
- validation rows
- remainder rows
- split method
- preserved column names

## Why this is safe

This step only reads `train.csv` and writes a CSV split. It does not:

- load or run the model;
- train/SFT;
- mutate the Golden adapter;
- write `adapter_model.safetensors`;
- mutate `adapter_config.json`;
- create `submission.zip`;
- submit to Kaggle.

## First-pass expected output

Because `predictions` and `logprobs` are intentionally left blank in the first pass:

- `category_map.csv` and `validation_set_labeled.csv` should be populated;
- Golden accuracy remains `NOT_AVAILABLE`;
- logprob analysis remains `NOT_AVAILABLE`;
- failure-type counts remain placeholder unless existing prediction artifacts are supplied.
