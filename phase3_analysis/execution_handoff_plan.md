# Phase3 Execution Handoff Plan

This plan separates work that can be done in this repository from work that must be run by the user inside Kaggle. The priority is to keep the Golden Baseline safe: no training, no adapter mutation, no `submission.zip`, and no Kaggle submission during Phase3 analysis.

## A. Work the assistant can do in this repository

These tasks do not require Kaggle GPU execution, competition data access, or Golden Baseline inference.

1. Maintain and refine the offline Phase3 analysis notebook/script.
   - Files: `phase3_offline_analysis.ipynb`, `phase3_make_recommendation.py`.
   - Scope: category classification, artifact schemas, summary generation, recommendation templates, reproducibility notes.

2. Prepare deterministic CV-design logic and documentation.
   - Define how to use Kaggle `train.csv` as the labeled source.
   - Recommend a fixed held-out split path such as `/kaggle/working/phase3_validation_split.csv`.
   - Keep prediction/logprob metrics as `NOT_AVAILABLE` if no Golden prediction artifact exists.

3. Provide schema checks and offline structural validation.
   - Compile Python script.
   - Compile embedded notebook code cell without running model inference.
   - Check CSV/JSONL headers and required artifact names.

4. Update reports after the user brings back Kaggle outputs.
   - If the user returns `phase3_analysis/` artifacts from Kaggle, inspect category maps, summaries, and recommendation outputs.
   - If prediction/logprob files are absent, interpret only category/CV-design outputs.
   - If prediction/logprob files are present later, compute measured failure summaries without changing Golden adapter settings.

5. Keep Golden logging as HOLD / low priority.
   - Do not add logging to the Golden Baseline notebook by default.
   - Only draft a separate logging-only runner if the user later decides measured Golden predictions are worth the operational risk.

## B. Work the user should run in Kaggle

These tasks require Kaggle competition data, Kaggle notebook execution, or the user's Golden Baseline environment.

1. Upload or open `phase3_offline_analysis.ipynb` in Kaggle with Internet OFF.
   - The notebook is self-contained and uses only Python standard library.
   - GPU is not required for category-map/CV-design mode, but the target environment can remain RTX Pro5000.

2. Create or choose the Phase3 validation file.
   - Preferred source: `/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv`.
   - Preferred derived validation path: `/kaggle/working/phase3_validation_split.csv`.
   - Set `create_validation_split=True` in `NOTEBOOK_CONFIG`, or run the equivalent CLI command in `validation_split_instructions.md`.
   - Use default seed `42` and default held-out fraction `0.2` unless you have a fixed prior convention.
   - Do not use Kaggle `test.csv` for local accuracy/failure analysis because gold answers are unavailable.

3. Configure `NOTEBOOK_CONFIG` in `phase3_offline_analysis.ipynb`.
   - Set `validation` to `/kaggle/working/phase3_validation_split.csv`.
   - Leave `predictions` and `logprobs` blank unless existing Golden artifacts already exist.
   - Keep `adapter_path` as the read-only Golden adapter path.

4. Run the notebook in category-map/CV-design mode first.
   - Expected measured accuracy/logprob status: `NOT_AVAILABLE` if no prediction/logprob artifacts are supplied.
   - Expected useful outputs: category map, labeled validation set, CV source documentation, reproducibility notes, and guarded next-experiment candidates.

5. Download or share the resulting `phase3_analysis/` directory.
   - Especially useful files: `category_map.csv`, `validation_set_labeled.csv`, `phase3_recommendation.md`, `run_commands.md`, and `reproducibility_notes.md`.
   - If no predictions were supplied, `golden_validation_summary.csv`, `category_failure_summary.csv`, and logprob/failure-type reports will remain placeholder/`NOT_AVAILABLE`.

6. Optional later step only if needed: provide existing Golden prediction/logprob artifacts.
   - Use existing files if they already exist outside the Golden Baseline notebook.
   - Do not modify the Golden Baseline just to produce logs in the first pass.
   - If a separate logging-only runner is eventually approved, it must preserve prompt, decoding, seed, temperature, max tokens, stop conditions, dtype, adapter, and model-loading behavior.

## C. Recommended first-pass workflow

1. Assistant keeps repository notebook/script ready and analysis-only.
2. User creates `/kaggle/working/phase3_validation_split.csv` from competition `train.csv` in Kaggle.
3. User runs `phase3_offline_analysis.ipynb` with `predictions=""` and `logprobs=""`.
4. User returns the generated `phase3_analysis/` outputs.
5. Assistant reviews the category distribution and CV-design artifacts.
6. Only after that, decide whether measured Golden prediction logging is worth moving from HOLD to ADOPT.

## D. Stop conditions

Stop and do not proceed if any step would:

- train or SFT a model;
- modify the Golden adapter;
- write or overwrite `adapter_model.safetensors`;
- mutate `adapter_config.json`;
- change LoRA rank, target modules, dtype, prompt, or decoding settings;
- create `submission.zip`;
- submit to Kaggle or use Public LB feedback as Phase3 evidence.
