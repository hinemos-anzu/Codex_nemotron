# Prompt3 static check report: B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT

## Change target

- Changed file: `experiments/b3_gate15_x17/b3_gate15_x17.py`.
- Golden source file not changed: `b3-nemotron-svd-26042701.ipynb`.
- Changed function / block: `build_b3_adapter()` Mamba merge loop, specifically `for layer_path, projs in sorted(mamba_merge_layers.items())`.

## Split change

- Before Prompt3: Golden copied merge block used `split_rank = 16` for both `gate_proj` and `x_proj`, producing `2 * split_rank = 32` merged `in_proj` rank.
- After Prompt3: copied merge block uses asymmetric component ranks:
  - `gate_proj` component rank: `GATE_COMPONENT_RANK = 15`
  - `x_proj` component rank: `X_COMPONENT_RANK = 17`
  - merged `in_proj` total rank: `IN_PROJ_TOTAL_RANK = 32`

## Evidence of gate-side rank constant

- The copied merge block assigns `gate_target_rank = GATE_COMPONENT_RANK`.
- The gate-side `compress_lora_fast(...)` call passes `gate_target_rank` as its target rank.
- The gate-side compression reason string was updated to `B3_gate_component_rank32_to_15_before_in_proj_merge`.
- The copied merge block asserts `gate_rank == GATE_COMPONENT_RANK` after compression.

## Evidence of x-side rank constant

- The copied merge block assigns `x_target_rank = X_COMPONENT_RANK`.
- The x-side `compress_lora_fast(...)` call passes `x_target_rank` as its target rank.
- The x-side compression reason string was updated to `B3_x_component_rank32_to_17_before_in_proj_merge`.
- The copied merge block asserts `x_rank == X_COMPONENT_RANK` after compression.

## Merged `in_proj` rank-32 rationale

- `GATE_COMPONENT_RANK + X_COMPONENT_RANK == IN_PROJ_TOTAL_RANK` is asserted near the experiment constants and again inside the merge block via `gate_target_rank + x_target_rank`.
- `merged_A = torch.cat([gate_A15, x_A17], dim=0)` makes the merged LoRA A rank dimension `15 + 17 = 32`.
- `merged_B = torch.zeros(in_proj_dim, IN_PROJ_TOTAL_RANK, ...)` explicitly allocates the merged LoRA B rank dimension as 32.
- Gate B columns occupy `:gate_rank` and x B columns occupy `gate_rank : gate_rank + x_rank`, preserving the Golden block-diagonal style with zero padding across the other component rank columns.
- The copied merge block asserts `merged_A.shape[0] == IN_PROJ_TOTAL_RANK` and `merged_B.shape[1] == IN_PROJ_TOTAL_RANK`.

## Added asserts / checks

- `assert gate_target_rank + x_target_rank == IN_PROJ_TOTAL_RANK`
- `assert gate_rank == GATE_COMPONENT_RANK`
- `assert x_rank == X_COMPONENT_RANK`
- `assert gate_rank + x_rank == IN_PROJ_TOTAL_RANK`
- `assert merged_A.shape[0] == IN_PROJ_TOTAL_RANK`
- `assert merged_B.shape[1] == IN_PROJ_TOTAL_RANK`

## RANK_MAP status

- `RANK_MAP` was not changed in Prompt3.
- Its `gate_proj` / `x_proj` entries remain 16 for Golden-reference readability, but the Mamba merge block no longer uses `RANK_MAP` or a fixed `split_rank = 16` for the merge compression. It explicitly uses `GATE_COMPONENT_RANK` and `X_COMPONENT_RANK`.

## Areas intentionally not changed

Prompt3 did not change:

- Golden source notebook
- Golden artifacts
- `q_proj` rank / processing
- `k_proj` rank / processing
- `v_proj` rank / processing
- `o_proj` rank / processing
- `up_proj` rank / processing
- `down_proj` rank / processing
- `out_proj` rank / processing
- `lm_head` rank / processing
- expert unfuse processing
- prefix conversion processing
- adapter_config generation logic
- zip generation logic
- input path
- output path
- dtype handling
- target_modules
- lora_alpha
- lora_dropout
- inference_mode

## Explicitly not executed / not generated

- Did not execute `build_b3_adapter()`.
- Did not execute the copied adapter/submission pipeline.
- Did not generate `adapter_model.safetensors`.
- Did not generate `adapter_config.json`.
- Did not create `submission.zip`.
- Did not run SFT/training.
- Did not submit to Kaggle.

## Static validation results

- PASS: `python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py`
- PASS: `git diff --name-only -- b3-nemotron-svd-26042701.ipynb` produced no output.
- PASS: `test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json`
- PASS: `git diff --check`
- PASS: `rg -n "split_rank\s*=" experiments/b3_gate15_x17/b3_gate15_x17.py || true` shows only a historical comment, not executable merge-block code.
- PASS: `find experiments -type d -name '__pycache__' -print` produced no output after cleanup.
