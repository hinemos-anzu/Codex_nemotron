# Kaggle Phase3 Empty Package Review

## Reviewed package

Reviewed `phase3_package (1).zip` from branch `codex/conduct-phase-3-analysis-for-nemotron-00nzc6`.

## Finding

This package is safe but not useful for category analysis because no validation data was supplied or created.

Evidence from `run_commands.md` inside the package:

| field | value |
|---|---|
| Validation path | `NOT_PROVIDED` |
| Prediction path | `NOT_PROVIDED` |
| Logprob path | `NOT_PROVIDED` |

Evidence from artifacts:

| artifact | status |
|---|---|
| `category_map.csv` | header only |
| `validation_set_labeled.csv` | header only |
| `golden_validation_predictions.jsonl` | empty |
| `golden_validation_summary.csv` | `n_total = 0`, accuracy `NOT_AVAILABLE` |

## Cause

The notebook was run with neither:

- `create_validation_split=True`, nor
- `validation` pointing to an existing CSV.

As a result, the analyzer had zero validation rows.

## Fix

Rerun the updated notebook with first-pass split creation enabled:

```python
NOTEBOOK_CONFIG = {
    "output_dir": "phase3_analysis",
    "create_validation_split": True,
    "train_source": "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv",
    "validation_output": "/kaggle/working/phase3_validation_split.csv",
    "validation_fraction": 0.2,
    "validation_size": None,
    "split_seed": "42",
    "validation": "",
    "predictions": "",
    "logprobs": "",
}
```

A successful first-pass rerun should again report:

- `validation_rows = 1900`;
- `prediction_rows = 0` is acceptable;
- `logprob_rows = 0` is acceptable;
- `phase3_validation_split.csv` and `phase3_validation_split.manifest.json` are included in the zip package.
