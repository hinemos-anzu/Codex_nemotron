# Baseline Adapter Inventory

```json
{
  "timestamp_utc": "2026-05-03T13:29:28.515230Z",
  "model_path": "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
  "adapter_path": "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
  "baseline_submission_path": "/kaggle/working/submission.zip",
  "baseline_inventory": {
    "error": "adapter path not found in this environment"
  },
  "zip_inventory": {
    "exists": false
  },
  "conversion_stage_log": [
    {
      "stage": "raw adapter load",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "prefix rename",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "expert unfuse",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "Mamba gate/x merge",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "SVD compression",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "target_modules reconciliation",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "adapter_config rewrite",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "final safetensors export",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    },
    {
      "stage": "submission.zip export",
      "status": "not_executed_in_this_environment",
      "notes": "instrumentation scaffold only"
    }
  ],
  "svd_error_report": {
    "status": "not_computed",
    "reason": "requires conversion run with original and compressed factors"
  }
}
```
