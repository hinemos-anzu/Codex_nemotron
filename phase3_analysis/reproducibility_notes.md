# Phase3 Reproducibility Notes

## Conditions for identical reruns

- Use the same Golden notebook/script, Golden adapter path, validation/held-out data, prediction log, logprob log, seed, and generation config.
- If generating predictions before this script, keep the Golden Baseline decoding settings unchanged.
- Do not alter LoRA rank, target modules, dtype, adapter files, or prompt templates during Phase3 analysis.

## Remaining nondeterminism

- This script is deterministic for fixed input files.
- Model generation outside this script may remain nondeterministic if CUDA kernels, sampling, vLLM scheduling, or temperature settings are nondeterministic.

## Logprob availability

- If token logprobs are present in the input logprob file, `min_logprob_summary.csv` and priority fields use them.
- If logprobs are unavailable, logprob columns are left blank and report sections mark confidence-based conclusions as unmeasured.

## vLLM vs transformers

- vLLM and transformers can differ in token accounting, finish reasons, and logprob APIs.
- Compare only runs created with the same inference backend unless the backend change itself is the isolated experiment variable.

## Kaggle Internet OFF compatibility

- The script uses only Python standard-library modules and local files, so it is suitable for Kaggle Internet OFF analysis.
- It does not clone Huikang pipeline code, download models, train, mutate adapters, zip submissions, or call Kaggle APIs.

## Fallback if inference/logprob extraction fails

1. Prefer an existing Golden prediction JSONL via `--predictions` if it already exists; do not create new Golden Baseline logging just for the first Phase3 pass.
2. Provide a held-out/validation CSV/JSONL via `--validation` for category labeling.
3. Rerun this script to compute summaries from available logs.
4. If logprobs remain unavailable, use failure-type counts and parse-error analysis only, and mark logprob conclusions as HOLD.
