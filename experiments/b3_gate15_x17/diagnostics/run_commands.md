# Golden artifact manifest run commands

Purpose: record the read-only discovery and manifest creation commands used before any experiment implementation.

## Commands run

```bash
pwd && rg --files -g 'AGENTS.md' -g '!**/.git/**' .. /workspace 2>/dev/null | head -50
find .. -name AGENTS.md -print
rg --files -g '!**/.git/**' | sed -n '1,200p'
find . -maxdepth 5 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_config.json' -o -name 'submission.zip' -o -name '*.ipynb' -o -name '*.py' \) -print
sed -n '1,220p' README.md
python - <<'PY'
import json, pathlib
p=pathlib.Path('reports/conversion_audit/baseline_adapter_inventory.json')
print(p.read_text()[:4000])
PY
python - <<'PY'
import json, pathlib, re
for p in pathlib.Path('.').glob('*.ipynb'):
    nb=json.loads(p.read_text())
    text='\n'.join(''.join(c.get('source','')) for c in nb.get('cells',[]))
    print('\n###', p)
    for pat in ['RANK_MAP','rank_map','target_modules','adapter_model','adapter_config','submission.zip','nemotron-adapter']:
        print(pat, text.find(pat))
PY
python - <<'PY'
import zipfile, pathlib
for p in pathlib.Path('.').glob('*.zip'):
    print('\n###',p)
    with zipfile.ZipFile(p) as z:
        for i in z.infolist()[:50]: print(i.filename, i.file_size)
        print('count', len(z.infolist()))
PY
python - <<'PY'
import zipfile,json
for zp in ['conversion_audit_reports.zip','conversion_audit_reports (1).zip','conversion_audit_reports (2).zip']:
 with zipfile.ZipFile(zp) as z:
  data=json.loads(z.read('baseline_adapter_inventory.json'))['report']
  print('\n',zp)
  def walk(o,path=''):
   if isinstance(o,dict):
    for k,v in o.items():
     if 'sha' in k.lower() or 'hash' in k.lower() or 'zip' in k.lower() or 'path' in k.lower(): print(path+k, type(v).__name__, v if not isinstance(v,(dict,list)) else '')
     walk(v,path+k+'.')
   elif isinstance(o,list):
    for i,v in enumerate(o[:3]): walk(v,path+str(i)+'.')
  walk(data)
PY
python - <<'PY'
# Manifest creation script (see shell history / committed diff for exact generated output)
PY
python -m json.tool experiments/b3_gate15_x17/diagnostics/golden_artifact_manifest.json >/tmp/golden_manifest_check.json
```

## Explicitly not run

```bash
# No adapter generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt2 commands run

```bash
git status --short
find .. -name AGENTS.md -print
python - <<'PY'
import json, pathlib
nb=json.loads(pathlib.Path('b3-nemotron-svd-26042701.ipynb').read_text())
print(len(nb.get('cells',[])))
for i,c in enumerate(nb.get('cells',[])):
    src=''.join(c.get('source',''))
    if any(s in src for s in ['merge_gate_x_to_in_proj','gate_proj','x_proj','compress_lora_fast','RANK_MAP']):
        print('CELL',i,c.get('cell_type'),len(src),src[:120].replace('\n',' '))
PY
python - <<'PY'
import json,pathlib
code=''.join(json.loads(pathlib.Path('b3-nemotron-svd-26042701.ipynb').read_text())['cells'][1]['source'])
for i,line in enumerate(code.splitlines(),1):
    if any(tok in line for tok in ['RANK_MAP','gate_proj','x_proj','compress_lora_fast','merge','in_proj','main()']):
        print(f'{i:04d}: {line}')
PY
python - <<'PY'
# Extract Golden notebook code cell to experiments/b3_gate15_x17/b3_gate15_x17.py,
# add Prompt2 constants and patch-plan comments only, and do not execute the script.
PY
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
python -m json.tool experiments/b3_gate15_x17/diagnostics/golden_artifact_manifest.json >/tmp/golden_manifest_check_prompt2.json
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
git diff --check
find experiments -type d -name '__pycache__' -print
```

## Prompt2 explicitly not run

```bash
# No execution of experiments/b3_gate15_x17/b3_gate15_x17.py
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt3 commands run

```bash
sed -n '360,430p' experiments/b3_gate15_x17/b3_gate15_x17.py
python - <<'PY'
# Static edit only: replaced the copied Mamba merge block's fixed split_rank=16
# with GATE_COMPONENT_RANK=15, X_COMPONENT_RANK=17, IN_PROJ_TOTAL_RANK=32,
# added shape/rank asserts, and did not execute build_b3_adapter().
PY
rg -n "split_rank\s*=\s*16|gate_B16|x_B16|B3_split_gate_x_16_16" experiments/b3_gate15_x17/b3_gate15_x17.py || true
rg -n "GATE_COMPONENT_RANK|X_COMPONENT_RANK|IN_PROJ_TOTAL_RANK|gate_target_rank|x_target_rank|merged_A.shape|merged_B.shape" experiments/b3_gate15_x17/b3_gate15_x17.py
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
git diff --check
rg -n "split_rank\s*=" experiments/b3_gate15_x17/b3_gate15_x17.py || true
find experiments -type d -name '__pycache__' -print
```

## Prompt3 explicitly not run

```bash
# No execution of experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt4 preflight commands run

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

Input paths checked:

- Adapter input: `/kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`
- Base model input: `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`

Output paths intended:

- Adapter output: `experiments/b3_gate15_x17/adapter/`
- Submission zip output: `experiments/b3_gate15_x17/submission.zip`
- Diagnostics output: `experiments/b3_gate15_x17/diagnostics/`

```bash
git status --short
find .. -name AGENTS.md -print
test -f experiments/b3_gate15_x17/b3_gate15_x17.py && echo target_ok
if test -d /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20; then echo adapter_input_ok; else echo adapter_input_missing; fi
if test -d /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1; then echo base_model_input_ok; else echo base_model_input_missing; fi
df -h . /tmp 2>/dev/null
sed -n '29,41p' experiments/b3_gate15_x17/b3_gate15_x17.py
python - <<'PY'
# I/O safety-only edit: changed copied script defaults from /kaggle/working/*
# to experiments/b3_gate15_x17/{adapter,diagnostics,submission.zip}.
PY
mkdir -p experiments/b3_gate15_x17/adapter experiments/b3_gate15_x17/diagnostics
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
if test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json; then echo kaggle_working_root_clean; else echo kaggle_working_root_polluted; fi
find experiments/b3_gate15_x17/adapter -maxdepth 1 -type f -print
git diff --check
```

Prompt4 STOP reason:

```text
adapter_input_missing
base_model_input_missing
```

## Prompt4 explicitly not run

```bash
# No execution of python experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
# No sha256 calculation for candidate artifacts because candidate artifacts were not generated
# No zip entry inspection for candidate submission.zip because candidate submission.zip was not generated
```

## Prompt4 rerun preflight commands run

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
git status --short
test -f experiments/b3_gate15_x17/b3_gate15_x17.py && echo target_ok
if test -d /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20; then echo adapter_input_ok; else echo adapter_input_missing; fi
if test -d /kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1; then echo base_model_input_ok; else echo base_model_input_missing; fi
df -h . /tmp 2>/dev/null
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
mkdir -p experiments/b3_gate15_x17/adapter experiments/b3_gate15_x17/diagnostics
find experiments/b3_gate15_x17/adapter -maxdepth 1 -type f -print
git diff --check
```

Prompt4 rerun STOP reason remains:

```text
adapter_input_missing
base_model_input_missing
```

## Prompt4 rerun explicitly not run

```bash
# No execution of python experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt4-Kaggle runbook preparation

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
cat > experiments/b3_gate15_x17/diagnostics/kaggle_prompt4_runbook.md <<'RUNBOOK'
# Prompt4-Kaggle runbook content for Kaggle Notebook execution
RUNBOOK
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Prompt4-Kaggle runbook preparation explicitly did **not** run:

```bash
# No execution of python experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt4-Kaggle all-in-one notebook creation

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
python - <<'PY'
# Read experiments/b3_gate15_x17/b3_gate15_x17.py and generate
# experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt4.ipynb
# as a Kaggle Internet OFF all-in-one notebook.
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt4.ipynb')
nb=json.loads(p.read_text())
text=p.read_text()
print('cells', len(nb['cells']))
for term in ['pip install','git clone','requests.get','wget','curl','kagglehub']:
    print(term, term in text)
for term in ['GATE_COMPONENT_RANK = 15','X_COMPONENT_RANK = 17','IN_PROJ_TOTAL_RANK = 32','SOURCE_GOLDEN_NOTEBOOK','SOURCE_PATCHED_SCRIPT','/kaggle/working/experiments/b3_gate15_x17/submission.zip']:
    print(term, term in text)
PY
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Notebook creation explicitly did **not** run:

```bash
# No execution of the generated notebook's candidate-generation cells
# No execution of python experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt4-Kaggle notebook syntax hotfix

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt4.ipynb')
nb=json.loads(p.read_text())
for i,c in enumerate(nb['cells'],1):
    if c['cell_type']=='code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
PY
python - <<'PY'
# Fixed escaped-newline string literals in the generated notebook utilities/report cells.
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt4.ipynb')
nb=json.loads(p.read_text())
text=p.read_text()
assert len(nb['cells']) == 11
for i,c in enumerate(nb['cells'],1):
    if c['cell_type']=='code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
for term in ['pip install','git clone','requests.get','wget','curl','kagglehub']:
    assert term not in text, term
for term in ['GATE_COMPONENT_RANK = 15','X_COMPONENT_RANK = 17','IN_PROJ_TOTAL_RANK = 32','SOURCE_GOLDEN_NOTEBOOK','SOURCE_PATCHED_SCRIPT','/kaggle/working/experiments/b3_gate15_x17/submission.zip']:
    assert term in text, term
print('notebook validation passed')
PY
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Notebook syntax hotfix explicitly did **not** run:

```bash
# No execution of the generated notebook's candidate-generation cells
# No execution of python experiments/b3_gate15_x17/b3_gate15_x17.py
# No build_b3_adapter() execution
# No adapter generation
# No adapter_config.json generation
# No submission.zip creation
# No SFT/training
# No Kaggle submission
```

## Prompt5 validation notebook creation

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
python - <<'PY'
# Generate experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt5_validation.ipynb
# containing Prompt5A lightweight checks and Prompt5B full safetensors checks.
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt5_validation.ipynb')
nb=json.loads(p.read_text())
for i,c in enumerate(nb['cells'],1):
    if c['cell_type']=='code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
print('prompt5 notebook code cells compile', len(nb['cells']))
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt5_validation.ipynb')
nb=json.loads(p.read_text())
text=p.read_text()
assert len(nb['cells']) == 7
for term in ['validation_precheck.json','adapter_consistency_check.json','submission_zip_check.json','diagnostic_summary.json','diff_report.md','rollback.md']:
    assert term in text, term
for term in ['copy_to_root_not_performed','kaggle_submit_not_performed','training_not_performed']:
    assert term in text, term
print('prompt5 notebook validation passed')
PY
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Prompt5 validation notebook creation explicitly did **not** run:

```bash
# No execution of the generated Prompt5 validation notebook
# No safetensors full scan in local Codex
# No adapter generation
# No adapter_config.json generation
# No submission.zip generation or regeneration
# No copy to /kaggle/working/submission.zip
# No SFT/training
# No Kaggle submission
```

## Prompt5A missing-artifact STOP clarification

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
python - <<'PY'
# Updated Prompt5 validation notebook so missing candidate artifacts produce
# a clearer STOP_REPORT_PROMPT5A with likely cause and recovery steps.
PY
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/b3_gate15_x17/b3_gate15_x17_kaggle_prompt5_validation.ipynb')
nb=json.loads(p.read_text())
for i,c in enumerate(nb['cells'],1):
    if c['cell_type']=='code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
text=p.read_text()
for term in ['likely_cause','recovery_steps','forbidden_recovery_actions','Do not copy to /kaggle/working/submission.zip yet.']:
    assert term in text, term
print('prompt5 missing-artifact clarification validation passed')
PY
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Prompt5A STOP clarification explicitly did **not** run:

```bash
# No execution of the generated Prompt5 validation notebook in local Codex
# No safetensors scan in local Codex
# No adapter generation
# No adapter_config.json generation
# No submission.zip generation or regeneration
# No copy to /kaggle/working/submission.zip
# No SFT/training
# No Kaggle submission
```

## Prompt5 audit evaluation report creation

Execution datetime: 2026-06-08 UTC in the local `/workspace/Codex_nemotron` environment.

```bash
cat > experiments/b3_gate15_x17/diagnostics/prompt5_evaluation_for_audit.md <<'EOF_REPORT'
# Prompt5 Evaluation Report for Audit Agent
EOF_REPORT
python -m py_compile experiments/b3_gate15_x17/b3_gate15_x17.py
rm -rf experiments/b3_gate15_x17/__pycache__
git diff --name-only -- b3-nemotron-svd-26042701.ipynb
test ! -e submission.zip && test ! -e adapter_model.safetensors && test ! -e adapter_config.json
test ! -e /kaggle/working/submission.zip && test ! -e /kaggle/working/adapter_model.safetensors && test ! -e /kaggle/working/adapter_config.json
git diff --check
```

Prompt5 audit report creation explicitly did **not** run:

```bash
# No execution of Kaggle Prompt4 or Prompt5 notebooks in local Codex
# No safetensors scan in local Codex
# No adapter generation
# No adapter_config.json generation
# No submission.zip generation or regeneration
# No copy to /kaggle/working/submission.zip
# No SFT/training
# No Kaggle submission
```
