# Prompt5 Evaluation Report for Audit Agent

## Executive summary

- **Prompt5A status: PASS**
- **Prompt5B status: PASS**
- **Overall Prompt5 status: PASS**
- **Candidate generation status before validation: SUCCESS**
- **Safe to proceed to Prompt6 copy-to-root only after explicit user approval: YES**
- **Kaggle Submit performed: NO**
- **Copy-to-root performed: NO**
- **SFT/training performed: NO**

This report summarizes the user-provided Kaggle logs for Prompt4 generation followed by Prompt5A/Prompt5B validation. The logs show the candidate adapter and candidate `submission.zip` were generated under `/kaggle/working/experiments/b3_gate15_x17/`, then validated successfully without copying to `/kaggle/working/submission.zip` and without Kaggle submission.

## Candidate artifact paths

| Artifact | Path | Size bytes | SHA256 |
| --- | --- | ---: | --- |
| adapter model | `/kaggle/working/experiments/b3_gate15_x17/adapter/adapter_model.safetensors` | `2468158128` | `d472f973a6a86d133539df0e849fb066e842963dfd18ea6c5476cb2a99a92040` |
| adapter config | `/kaggle/working/experiments/b3_gate15_x17/adapter/adapter_config.json` | `861` | `b9ea8cc87840e960954ab50ffcbcc23753b60fd5ce5e967f4e0657d44135fcbc` |
| candidate ZIP | `/kaggle/working/experiments/b3_gate15_x17/submission.zip` | `2280844012` | `9622a4bde86a72d771f3fcee1f891c200f8d8e9ff9f89133725999cf73e00574` |

## Prompt4 generation evidence from log

The log shows Prompt4 preflight passed with:

- `golden_adapter_dir_exists = true`
- `golden_adapter_model_exists = true`
- `golden_adapter_config_exists = true`
- `base_model_dir_exists = true`
- `base_model_safetensors_count = 13`
- `output_under_experiment_dir = true`
- `rank_assert_ok = true`
- `root_pollution_check.ok = true`
- `existing_direct_working_artifacts = []`

Prompt4 then generated:

- candidate adapter model at `/kaggle/working/experiments/b3_gate15_x17/adapter/adapter_model.safetensors`
- candidate config at `/kaggle/working/experiments/b3_gate15_x17/adapter/adapter_config.json`
- candidate zip at `/kaggle/working/experiments/b3_gate15_x17/submission.zip`

Prompt4 final summary recorded:

- `status: SUCCESS`
- `gate=15 / x=17 / in_proj rank=32`
- `SFT/training: not performed`
- `Kaggle Submit: not performed`

## Prompt4 adapter validation evidence

The Prompt4 adapter validation block reported:

| Check | Result | Audit interpretation |
| --- | --- | --- |
| `missing_target_modules` | `[]` | PASS |
| `rank_violations_rank_gt_32` | `[]` | PASS |
| `max_rank_seen` | `32` | PASS |
| `num_tensors` | `12010` | PASS |
| `contains_base_model_model_model` | `false` | PASS |
| `contains_base_model_model_backbone` | `true` | PASS |
| `bad_pattern_counts` | `{}` | PASS |
| `zero_shape_tensor_count` | `0` | PASS |
| `suffix_counts.in_proj` | `23` | PASS |

## ZIP validation evidence

The candidate ZIP check reported:

- `expected_exact_entries_present = true`
- `unexpected_entries = []`
- ZIP entries:
  - `adapter_model.safetensors`
  - `adapter_config.json`

The candidate ZIP contains exactly the two expected files and no diagnostics/log/notebook/cache/git/csv/jsonl artifacts.

## Prompt5A lightweight validation evidence

Prompt5A reported `Prompt5A PASS`.

Key Prompt5A checks:

| Check | Result |
| --- | --- |
| adapter_model manifest size match | `true` |
| adapter_model manifest sha256 match | `true` |
| adapter_config manifest size match | `true` |
| adapter_config manifest sha256 match | `true` |
| submission_zip manifest size match | `true` |
| submission_zip manifest sha256 match | `true` |
| ZIP config matches local config | `true` |
| ZIP model size matches local model | `true` |
| ZIP model sha256 matches local model | `true` |
| root pollution check | `ok = true` |
| failures | `[]` |

### Prompt5A adapter_config checks

| adapter_config field | Result |
| --- | --- |
| `peft_type == LORA` | `true` |
| `r == 32` | `true` |
| `lora_alpha == 32` | `true` |
| `lora_dropout == 0` | `true` |
| `inference_mode == true` | `true` |
| required target modules present | `true` |
| `gate_proj` / `x_proj` excluded | `true` |

Target modules recorded by Prompt5A:

```text
k_proj, o_proj, in_proj, q_proj, up_proj, v_proj, down_proj, out_proj, lm_head
```

## Prompt5B full safetensors validation evidence

Prompt5B reported `Prompt5B PASS` and `overall_status = PASS`.

| Prompt5B metric | Result | Audit interpretation |
| --- | ---: | --- |
| `num_tensors` | `12010` | PASS |
| `num_lora_pairs` | `6005` | PASS |
| `max_rank_seen` | `32` | PASS |
| rank violations | `[]` | PASS |
| NaN count | `0` | PASS |
| Inf count | `0` | PASS |
| prefix contains backbone | `True` | PASS |
| prefix contains model.model | `False` | PASS |
| forbidden patterns | `{}` | PASS |
| `in_proj_module_count` | `23` | PASS |
| `in_proj_rank_distribution` | `{'32': 23}` | PASS |
| ZIP entries | `adapter_model.safetensors`, `adapter_config.json` | PASS |
| `safe_to_copy_to_root` | `True` | PASS for Prompt6 eligibility |

## Gate/X split evidence

Prompt5B logged the following gate/x evidence:

```json
{
  "candidate_manifest_gate_component_rank": 15,
  "candidate_manifest_x_component_rank": 17,
  "candidate_manifest_in_proj_total_rank": 32,
  "svd_log_exists": true,
  "gate_rank15_log_count": 23,
  "x_rank17_log_count": 23,
  "static_report_exists": false,
  "generation_report_exists": true
}
```

Audit interpretation:

- Candidate manifest records `gate_component_rank = 15`, `x_component_rank = 17`, and `in_proj_total_rank = 32`.
- Compression log exists.
- `gate_rank15_log_count = 23` and `x_rank17_log_count = 23`, matching the expected 23 Mamba merge layers.
- Final `in_proj` rank distribution is `{'32': 23}`, confirming every merged `in_proj` module has rank 32.

## Golden protection / no-touch status

Golden artifacts remain protected based on the workflow and logs:

- Golden inputs are under `/kaggle/input/...`, which is read-only in Kaggle.
- Candidate outputs are under `/kaggle/working/experiments/b3_gate15_x17/`.
- Direct `/kaggle/working` root artifact pollution check is `ok = true`.
- `copy_to_root_not_performed = true`.
- `kaggle_submit_not_performed = true`.
- `training_not_performed = true`.

Note: Prompt5A reported `golden_manifest_exists = false` in the Kaggle diagnostics directory. This is not a candidate validation failure because the candidate artifact, ZIP, config, rank, prefix, and tensor checks all passed. For stronger audit reproducibility, the Golden manifest can be copied into the Kaggle working diagnostics directory in future runs.

## PASS / HOLD / FAIL decision

**Prompt5 audit decision: PASS**

Rationale:

- Prompt5A lightweight artifact/config/ZIP/manifest checks passed.
- Prompt5B full safetensors checks passed.
- Required rank, prefix, in_proj, NaN/Inf, forbidden-pattern, target module, and ZIP checks all passed.
- Candidate zip is marked safe to copy to root by validation logic.
- No training, copy-to-root, or Kaggle Submit occurred.

## Remaining restrictions

Even though Prompt5 passed, the following remain prohibited until explicit user approval:

- Do not Kaggle Submit.
- Do not modify Golden artifacts.
- Do not regenerate the candidate adapter or ZIP unless a new experiment is requested.

## Recommended next step

Proceed to **Prompt6: copy-to-root** only after explicit user approval.

Prompt6 action:

```text
Copy from:
/kaggle/working/experiments/b3_gate15_x17/submission.zip

Copy to:
/kaggle/working/submission.zip
```

Prompt6 must verify:

- `diagnostic_summary.json.validation_passed = true`
- `diagnostic_summary.json.safe_to_copy_to_root = true`
- source zip exists
- copied root zip sha256 and size match source zip
- Kaggle Submit is still not performed

Kaggle Submit should be a separate later step requiring explicit approval.
