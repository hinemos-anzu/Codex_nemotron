# Prompt5 validation notebook creation report

## Created artifact

- Notebook path: `experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt5_validation.ipynb`
- Purpose: Kaggle read-only validation notebook for Prompt5A and Prompt5B after Prompt4 candidate generation.
- Local Codex action: notebook creation only; no candidate artifacts were read or generated locally.

## Notebook scope

The notebook contains two validation stages:

1. Prompt5A lightweight validation:
   - candidate artifact existence
   - sha256 / size recomputation
   - candidate manifest comparison
   - ZIP entries check
   - adapter_config semantic checks
   - root pollution check
   - writes `validation_precheck.json` and `validation_precheck_report.md`

2. Prompt5B full tensor validation:
   - read-only safetensors scan
   - tensor count / dtype / shape distributions
   - NaN / Inf / zero-shape checks
   - LoRA pair and rank checks
   - prefix and forbidden-pattern checks
   - `in_proj` module count and rank distribution checks
   - target_modules consistency
   - ZIP recheck
   - writes `adapter_consistency_check.json`, `submission_zip_check.json`, `diagnostic_summary.json`, `diff_report.md`, and `rollback.md`

## Safety properties

- The notebook does not generate adapter artifacts.
- The notebook does not regenerate `submission.zip`.
- The notebook does not copy to `/kaggle/working/submission.zip`.
- The notebook does not run SFT/training.
- The notebook does not submit to Kaggle.
- Candidate and Golden artifacts are treated as read-only.

## Local validation performed

- Notebook JSON loads successfully.
- Notebook has 7 cells.
- All code cells compile with Python `compile(...)`.
- Required Prompt5A / Prompt5B output filenames are present.
- Required safety strings are present: `copy_to_root_not_performed`, `kaggle_submit_not_performed`, and `training_not_performed`.
- Repo-root artifact absence check passed.
- Direct `/kaggle/working` artifact absence check passed.
- Golden notebook diff check produced no output.

## Not performed locally

- Did not execute the generated Prompt5 validation notebook.
- Did not scan any Kaggle candidate safetensors locally.
- Did not generate `adapter_model.safetensors`.
- Did not generate `adapter_config.json`.
- Did not generate or regenerate `submission.zip`.
- Did not copy candidate zip to `/kaggle/working/submission.zip`.
- Did not run SFT/training.
- Did not submit to Kaggle.
