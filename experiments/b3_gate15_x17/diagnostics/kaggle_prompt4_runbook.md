# Prompt4-Kaggle runbook: candidate adapter/submission generation

## Status

Local Codex Prompt4 stopped correctly because the Kaggle input adapter and base model paths are not present in this environment. Do **not** proceed to Prompt5 until Prompt4 is completed in a Kaggle Notebook environment and candidate artifacts are produced.

## Purpose

Run `experiments/b3_gate15_x17/b3_gate15_x17.py` on Kaggle to generate the B3 gate=15 / x=17 candidate artifacts only:

- `experiments/b3_gate15_x17/adapter/adapter_model.safetensors`
- `experiments/b3_gate15_x17/adapter/adapter_config.json`
- `experiments/b3_gate15_x17/submission.zip`
- diagnostics logs / manifest / report under `experiments/b3_gate15_x17/diagnostics/`

Do not run SFT/training. Do not submit to Kaggle.

## Required input paths on Kaggle

Both paths must exist before generation starts:

```text
/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20
/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1
```

## Required output paths

The copied script defaults should already point under the experiment directory:

```text
WORKING_ADAPTER_DIR = experiments/b3_gate15_x17/adapter
DIAG_DIR = experiments/b3_gate15_x17/diagnostics
SUBMISSION_ZIP = experiments/b3_gate15_x17/submission.zip
```

Do not write candidate artifacts to repo root or `/kaggle/working` directly.

## Kaggle preflight cell

Run this first in Kaggle:

```python
from pathlib import Path

paths = [
    "/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20",
    "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1",
    "experiments/b3_gate15_x17/b3_gate15_x17.py",
]

for p in paths:
    print(p, Path(p).exists())
```

Proceed only if all checks print `True`.

## Kaggle execution cell

Run candidate generation with stdout/stderr captured:

```bash
mkdir -p experiments/b3_gate15_x17/adapter experiments/b3_gate15_x17/diagnostics
python experiments/b3_gate15_x17/b3_gate15_x17.py \
  > experiments/b3_gate15_x17/diagnostics/generation_stdout.log \
  2> experiments/b3_gate15_x17/diagnostics/generation_stderr.log
```

## Artifact verification and manifest cell

After generation, verify existence, sizes, hashes, and ZIP entries:

```python
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, UTC

experiment_name = "B3_GATE15_X17_ASYMMETRIC_INPROJ_SPLIT"
script_path = Path("experiments/b3_gate15_x17/b3_gate15_x17.py")
adapter_dir = Path("experiments/b3_gate15_x17/adapter")
diag_dir = Path("experiments/b3_gate15_x17/diagnostics")
submission_zip = Path("experiments/b3_gate15_x17/submission.zip")

adapter_model = adapter_dir / "adapter_model.safetensors"
adapter_config = adapter_dir / "adapter_config.json"
required = [adapter_model, adapter_config, submission_zip]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(missing)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def artifact_info(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

with zipfile.ZipFile(submission_zip, "r") as zf:
    zip_entries = [
        {
            "name": info.filename,
            "file_size": info.file_size,
            "compress_size": info.compress_size,
            "crc": info.CRC,
        }
        for info in zf.infolist()
    ]

entry_names = sorted(row["name"] for row in zip_entries)
if entry_names != ["adapter_config.json", "adapter_model.safetensors"]:
    raise ValueError(f"Unexpected ZIP entries: {entry_names}")

manifest = {
    "experiment_name": experiment_name,
    "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "script_path": str(script_path),
    "adapter_dir": str(adapter_dir),
    "submission_zip_path": str(submission_zip),
    "adapter_model": artifact_info(adapter_model),
    "adapter_config": artifact_info(adapter_config),
    "submission_zip": artifact_info(submission_zip),
    "zip_entries": zip_entries,
    "gate_component_rank": 15,
    "x_component_rank": 17,
    "in_proj_total_rank": 32,
    "training_performed": False,
    "kaggle_submit_performed": False,
}

manifest_path = diag_dir / "candidate_artifact_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
```

## Generation report cell

Create the required report after the manifest cell succeeds:

```python
import json
from pathlib import Path

manifest_path = Path("experiments/b3_gate15_x17/diagnostics/candidate_artifact_manifest.json")
manifest = json.loads(manifest_path.read_text())
report_path = Path("experiments/b3_gate15_x17/diagnostics/generation_report.md")

zip_entries = "\n".join(f"- {row['name']} ({row['file_size']} bytes)" for row in manifest["zip_entries"])

report_path.write_text(f'''# Prompt4-Kaggle generation report

## Execution target

- Script: `{manifest['script_path']}`
- Environment: Kaggle Notebook
- Input adapter path: `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`
- Input base model path: `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`

## Output paths

- Adapter dir: `{manifest['adapter_dir']}`
- Submission zip: `{manifest['submission_zip_path']}`
- Diagnostics dir: `experiments/b3_gate15_x17/diagnostics`

## Gate/x split

- gate_component_rank = {manifest['gate_component_rank']}
- x_component_rank = {manifest['x_component_rank']}
- in_proj_total_rank = {manifest['in_proj_total_rank']}

## Generated artifacts

- adapter_model.safetensors: `{manifest['adapter_model']['path']}` / {manifest['adapter_model']['size_bytes']} bytes / `{manifest['adapter_model']['sha256']}`
- adapter_config.json: `{manifest['adapter_config']['path']}` / {manifest['adapter_config']['size_bytes']} bytes / `{manifest['adapter_config']['sha256']}`
- submission.zip: `{manifest['submission_zip']['path']}` / {manifest['submission_zip']['size_bytes']} bytes / `{manifest['submission_zip']['sha256']}`

## ZIP entries

{zip_entries}

## Safety assertions

- Golden notebook was not edited.
- Golden input adapter was treated as read-only.
- SFT/training was not performed.
- Kaggle Submit was not performed.

## Prompt5 validation checklist

- Verify all LoRA tensor ranks, especially merged `in_proj` rank 32.
- Verify gate/x split evidence in tensor shapes and compression logs.
- Verify no NaN/Inf tensors.
- Verify prefix conversion and target_modules.
- Verify ZIP contains exactly `adapter_model.safetensors` and `adapter_config.json`.
- Compare candidate manifest against Golden manifest before any submission.
''', encoding="utf-8")
print(report_path)
```

## Forbidden operations

Do not run:

```bash
# No SFT/training
# No kaggle competitions submit
# No git add for adapter_model.safetensors or submission.zip
# No writes to repo-root submission.zip / adapter_model.safetensors / adapter_config.json
# No writes to /kaggle/working/submission.zip or direct /kaggle/working artifact files
```
