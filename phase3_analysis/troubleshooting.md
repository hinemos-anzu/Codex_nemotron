# Phase3 Offline Notebook Troubleshooting

This file lists the first checks to run when `phase3_offline_analysis.ipynb` fails in Kaggle.

## 1. If the error is `FileNotFoundError` for `train.csv`

Most likely the Kaggle competition dataset is not attached to the notebook, or the path in `NOTEBOOK_CONFIG["train_source"]` is different from the actual mounted path.

Expected default path:

```text
/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv
```

Kaggle-side checks:

```python
from pathlib import Path
print(Path('/kaggle/input').exists())
print([p.name for p in Path('/kaggle/input').iterdir()])
print(Path('/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv').exists())
```

Fix:

1. In Kaggle, click **Add Data**.
2. Add the competition dataset: `nvidia-nemotron-model-reasoning-challenge`.
3. Re-run the check above.
4. If Kaggle mounted it under a different folder name, update `NOTEBOOK_CONFIG["train_source"]` to the actual path.

## 2. If the notebook runs but `category_map.csv` is empty

Check whether `create_validation_split` is enabled or `validation` points to an existing CSV.

Recommended first-pass config:

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
}
```

Expected populated outputs:

- `/kaggle/working/phase3_validation_split.csv`
- `/kaggle/working/phase3_validation_split.manifest.json`
- `phase3_analysis/category_map.csv`
- `phase3_analysis/validation_set_labeled.csv`

## 3. If `validation_rows` is `0`

This means the notebook did not receive a validation CSV and did not create one from `train.csv`. In the uploaded `phase3_package (1).zip`, `run_commands.md` showed `Validation path: NOT_PROVIDED`, and `category_map.csv` contained only the header. That run is safe, but it is not useful for category analysis.

Fix the `NOTEBOOK_CONFIG` block and rerun:

```python
"create_validation_split": True,
"train_source": "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv",
"validation_output": "/kaggle/working/phase3_validation_split.csv",
"validation": "",
"predictions": "",
"logprobs": "",
```

If your Kaggle mount path differs, update `train_source` to the actual path shown under `/kaggle/input`. A successful first-pass run should report `validation_rows: 1900` for the 20% split of the 9500-row train file.

## 4. If Golden accuracy is `NOT_AVAILABLE`

This is expected in the first pass when `predictions` and `logprobs` are blank. The first pass is category-map/CV-design only and intentionally does not modify Golden Baseline logging.

## 5. If you see `fatal: not a git repository` after successful output

This message is benign in Kaggle when the notebook is not running inside a Git checkout. The Phase3 script only tries to record a git hash in `run_commands.md`; if no `.git` directory exists, the hash is recorded as `NOT_AVAILABLE`.

The notebook run is still successful if the JSON manifest was printed and the safety block reports:

```json
"training_executed": false,
"adapter_mutated": false,
"submission_zip_created": false,
"kaggle_submit_executed": false
```

The script now suppresses this Git stderr message and silently records `NOT_AVAILABLE` for the git hash outside a repository.

## 6. If you need help debugging

Please share the traceback text, especially:

- exception type;
- the last 20 lines of the stack trace;
- current `NOTEBOOK_CONFIG` values;
- output of the `/kaggle/input` path check above.
