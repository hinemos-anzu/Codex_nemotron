# Prompt6 copy-to-root notebook creation report

## Created artifact

- Notebook path: `experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt6_copy_to_root.ipynb`
- Purpose: Kaggle copy-to-root notebook for copying the Prompt5-validated candidate ZIP to `/kaggle/working/submission.zip`.
- Local Codex action: notebook creation only; no local copy-to-root or Kaggle Submit was performed.

## Notebook scope

The notebook performs only Prompt6 copy-to-root:

1. Configures source and root ZIP paths.
2. Loads utility functions for SHA256 and ZIP entry checks.
3. Verifies `diagnostic_summary.json` exists and contains:
   - `validation_passed = true`
   - `safe_to_copy_to_root = true`
   - `kaggle_submit_not_performed = true`
   - `copy_to_root_not_performed = true`
4. Verifies source ZIP exists and contains exactly:
   - `adapter_model.safetensors`
   - `adapter_config.json`
5. Copies source ZIP to `/kaggle/working/submission.zip` using `shutil.copy2`.
6. Verifies copied root ZIP size, SHA256, and ZIP entries match source.
7. Writes:
   - `copy_to_root_check.json`
   - `copy_to_root_report.md`
   - run command entry in diagnostics `run_commands.md`

## Safety properties

- The notebook does not generate adapter artifacts.
- The notebook does not regenerate `submission.zip`.
- The notebook does not run SFT/training.
- The notebook does not call Kaggle Submit.
- The notebook does not create `/kaggle/working/adapter_model.safetensors`.
- The notebook does not create `/kaggle/working/adapter_config.json`.

## Local validation performed

- Notebook JSON loads successfully.
- Notebook has 6 cells.
- All code cells compile with Python `compile(...)`.
- Required safety/output strings are present.
- Source patched script syntax check passed.
- Golden notebook diff check produced no output.
- Repo-root artifact absence check passed.
- Direct `/kaggle/working` artifact absence check passed.

## Not performed locally

- Did not execute the generated Prompt6 notebook.
- Did not copy any ZIP to `/kaggle/working/submission.zip` locally.
- Did not generate adapter artifacts.
- Did not regenerate candidate ZIP.
- Did not run SFT/training.
- Did not submit to Kaggle.
