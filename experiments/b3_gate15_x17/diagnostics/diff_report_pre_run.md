# Prompt2 pre-run diff report: B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT

## Scope and safety status

- Prompt2 objective: identify the Golden Mamba `gate_proj` / `x_proj` to `in_proj` merge block and prepare a copied patch target only.
- Golden source file: `b3-nemotron-svd-26042701.ipynb`.
- Golden manifest: `experiments/b3_gate15_x17/diagnostics/golden_artifact_manifest.json`.
- Copy / patch-target file: `experiments/b3_gate15_x17/b3_gate15_x17.py`.
- The Golden notebook was read only and was not edited.
- The copied script was not executed.

## Mamba merge processing identified

The Golden code cell contains the Mamba merge flow below; the copied script preserves the same logic and adds Prompt2 patch-plan markers only.

1. `find_mamba_merge_layers(base_names)` discovers layer prefixes that contain both `.gate_proj` and `.x_proj` LoRA bases.
2. `build_b3_adapter()` computes `base_names` from adapter tensor keys, calls `find_mamba_merge_layers`, and prints `Mamba merge layers: {len(mamba_merge_layers)}`.
3. The standard tensor loop skips bases in `mamba_merge_bases` so `gate_proj` / `x_proj` are not emitted as standalone outputs.
4. A later loop over `mamba_merge_layers` reads the raw `gate_proj` and `x_proj` LoRA tensors, compresses each component, concatenates A factors, block-places B factors, and writes the merged result as `in_proj.lora_A.weight` / `in_proj.lora_B.weight`.

## Mamba merge layers

- Expected Mamba merge layers: **23**.
- Basis: the prior Golden audit recorded 46 `gate_proj` tensors and 46 `x_proj` tensors. Since each LoRA module has two tensors (`lora_A.weight` and `lora_B.weight`), this corresponds to 23 gate modules and 23 x modules, i.e. 23 complete merge layers.

## Current 16/16 split evidence

Current Golden behavior is 16/16-equivalent for the Mamba merge because:

- `RANK_MAP` contains `"gate_proj": 16` and `"x_proj": 16`.
- Inside the merge loop, `split_rank = 16` is set immediately after `raw_rank = gate_A.shape[0]`.
- `compress_lora_fast(..., split_rank, module=...gate_component...)` compresses the gate component to rank 16.
- `compress_lora_fast(..., split_rank, module=...x_component...)` compresses the x component to rank 16.
- `merged_A = torch.cat([gate_A16, x_A16], dim=0)` and `merged_B = torch.zeros(in_proj_dim, 2 * split_rank, ...)` create total in-proj rank 32.
- The compression log reason is `B3_split_gate_x_16_16_block_merge_to_in_proj_rank32`.

## Planned 15/17 split

Prompt2 added the following constants to the copied patch target, without wiring them into execution yet:

```python
EXPERIMENT_NAME = "B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT"
GATE_COMPONENT_RANK = 15
X_COMPONENT_RANK = 17
IN_PROJ_TOTAL_RANK = 32
assert GATE_COMPONENT_RANK + X_COMPONENT_RANK == IN_PROJ_TOTAL_RANK
```

Planned Prompt3 static edit in the copied file only:

- Use `GATE_COMPONENT_RANK` for the gate-side `compress_lora_fast` target rank.
- Use `X_COMPONENT_RANK` for the x-side `compress_lora_fast` target rank.
- Keep merged `in_proj` total rank at `IN_PROJ_TOTAL_RANK == 32`.
- Update variable names / tensor slicing in the merge block so asymmetric gate=15 / x=17 placement is explicit and auditable.

## Change locations in copied script

- Patch-plan constants: `experiments/b3_gate15_x17/b3_gate15_x17.py`, near the path configuration block.
- Discovery function: `find_mamba_merge_layers`.
- Mamba merge caller: `mamba_merge_layers, mamba_merge_bases = find_mamba_merge_layers(base_names)`.
- Current 16/16 split point: `split_rank = 16` inside the `for layer_path, projs in sorted(mamba_merge_layers.items())` loop.
- Gate compression call: `gate_B16, gate_A16 = compress_lora_fast(... split_rank ...)`.
- X compression call: `x_B16, x_A16 = compress_lora_fast(... split_rank ...)`.
- Merged `in_proj` write: `output_tensors[f"{in_proj_base}.lora_A.weight"]` and `output_tensors[f"{in_proj_base}.lora_B.weight"]`.

## Areas intentionally not changed

The Prompt2 copy / patch plan does **not** change:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `up_proj`
- `down_proj`
- `out_proj`
- `lm_head`
- expert unfuse
- prefix conversion
- adapter_config generation
- zip generation

## Explicitly not executed / not generated

- Did not execute the copied script.
- Did not generate `adapter_model.safetensors`.
- Did not generate `adapter_config.json`.
- Did not create `submission.zip`.
- Did not run SFT/training.
- Did not submit to Kaggle.
