# Notebook creation report: B3 Gate15/X17 Kaggle Prompt4

## Created artifact

- Notebook path: `experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt4.ipynb`
- Purpose: Kaggle Internet OFF all-in-one Prompt4 candidate generation notebook.
- Local Codex action: notebook creation only; no adapter or submission artifacts were generated locally.

## Source provenance recorded

The notebook records the following provenance in notebook metadata and Config cell:

- `source_golden_notebook = "b3-nemotron-svd-26042701.ipynb"`
- `source_patched_script = "experiments/b3_gate15_x17/b3_gate15_x17.py"`
- `source_diff_report = "experiments/b3_gate15_x17/diagnostics/diff_report_pre_run.md"`
- `source_static_check_report = "experiments/b3_gate15_x17/diagnostics/static_check_report.md"`

The notebook was generated from the copied patched script, not by editing the Golden notebook.

## Notebook structure

The notebook contains 11 cells:

1. Markdown title and safety scope.
2. Config and source provenance constants.
3. Imports and utilities.
4. Preflight checks with STOP report creation.
5. Full all-in-one B3 weight surgery code embedded in the notebook.
6. Candidate adapter generation cell.
7. Candidate `submission.zip` creation and ZIP entry check.
8. Candidate artifact manifest creation.
9. Generation report and run command logging.
10. Final summary print.
11. Disabled optional copy-to-root cell for a future user-authorized post-Prompt5 step.

## Internet OFF compatibility

The notebook is self-contained and does not depend on importing `experiments.b3_gate15_x17.b3_gate15_x17` or invoking the script with a shell command. The B3 code is embedded directly in notebook cells.

Validation confirmed the notebook text does not contain the prohibited network/setup tokens checked for this task:

- `pip install`
- `git clone`
- `requests.get`
- `wget`
- `curl`
- `kagglehub`

## Gate/X split included

The notebook includes:

- `GATE_COMPONENT_RANK = 15`
- `X_COMPONENT_RANK = 17`
- `IN_PROJ_TOTAL_RANK = 32`
- `assert GATE_COMPONENT_RANK + X_COMPONENT_RANK == IN_PROJ_TOTAL_RANK`

The embedded Mamba merge block uses `gate_target_rank = GATE_COMPONENT_RANK` and `x_target_rank = X_COMPONENT_RANK`, then asserts the merged `in_proj` rank is 32.

## Output safety

Kaggle output paths are fixed under:

- `BASE_OUT_DIR = Path("/kaggle/working/experiments/b3_gate15_x17")`
- `ADAPTER_OUT_DIR = BASE_OUT_DIR / "adapter"`
- `DIAG_DIR = BASE_OUT_DIR / "diagnostics"`
- `SUBMISSION_ZIP = BASE_OUT_DIR / "submission.zip"`

The notebook checks direct `/kaggle/working` artifact pollution before generation and refuses to overwrite direct root artifacts.

## Local validation performed

- Notebook JSON loaded successfully.
- Notebook has 11 cells.
- Prohibited token check passed for the notebook text.
- Required provenance constants are present.
- Required gate/x rank constants are present.
- Required safe candidate zip path `/kaggle/working/experiments/b3_gate15_x17/submission.zip` is present.
- Golden notebook diff check produced no output.
- Repo-root artifact absence check passed.
- Direct `/kaggle/working` artifact absence check passed.
- Python syntax check for the source patched script passed.

## Not performed locally

- Did not execute notebook cells that generate artifacts.
- Did not execute `build_b3_adapter()`.
- Did not generate `adapter_model.safetensors`.
- Did not generate `adapter_config.json`.
- Did not generate `submission.zip`.
- Did not run SFT/training.
- Did not submit to Kaggle.
