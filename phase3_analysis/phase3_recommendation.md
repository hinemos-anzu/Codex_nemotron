# Phase3 Analysis Report


## 0. Current Operating Priority

- Do **not** modify the Golden Baseline notebook just to add logging in the first pass.
- First option: build a deterministic `train.csv`-derived held-out split and run category-map/CV-design analysis.
- If no existing Golden prediction/logprob artifact is available, measured accuracy/failure/logprob sections remain `NOT_AVAILABLE`.
- A logging-only validation runner is HOLD / low priority and should be added only later if measured evidence is worth the Golden Baseline touch risk.

## 0.5 Handoff Summary

- Assistant/repo side: maintain the offline analysis notebook/script, schemas, documentation, and post-run interpretation.
- User/Kaggle side: create the deterministic `train.csv`-derived held-out split, run the self-contained notebook in Internet-OFF mode, and return the generated `phase3_analysis/` outputs.
- First pass should not add Golden Baseline logging; prediction/logprob measurements stay `NOT_AVAILABLE` unless existing artifacts are supplied.

## 1. Summary

### Facts from local artifacts

- Validation件数: `0`
- Golden accuracy: `NOT_AVAILABLE`
- Parse success rate: `NOT_AVAILABLE`
- Measurement status: `NOT_MEASURED: validation/prediction/logprob inputs were not present in this repository`

### Unconfirmed / not measured in this run

- Public LB 0.86 is treated as user-provided context only; this script did not submit to Kaggle or query the Public LB.
- Category weakness rankings are data-driven only when validation predictions are supplied.  Without those logs, the next-experiment list below is a guarded hypothesis framework, not a measured claim.

## 2. Category Failure Summary

| category | subcategory | n | accuracy | avg_min_logprob | n_wrong_low_conf | n_correct_low_conf | priority_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | NOT_AVAILABLE | 0 | NOT_AVAILABLE |  | 0 | 0 |  |

## 3. Cryptarithm Findings

- Main failure types are computed in `failure_cases_cryptarithm.csv` when cryptarithm rows and Golden predictions are available.
- Solver check possible: yes for well-formed alphametic/digit-assignment tasks.
- Synthetic generation possible: yes, but should be isolated into one-variable +100-example experiments.
- Recommended template: column-wise constraint table with explicit carry, leading-zero, and mapping-conflict checks.

## 4. Bit Manipulation Findings

- Main failure types are computed in `failure_cases_bit_manipulation.csv` when local rows are available.
- Recommended next candidate: XOR/base-conversion examples with operator-by-operator binary workspace and decimal cross-check.

## 5. Numeral Conversion Findings

- Main failure types are computed in `failure_cases_numeral_conversion.csv` when local rows are available.
- Recommended next candidate: place-value conversion examples with inverse conversion checks.

## 6. Protected Cases

- Protected cases are defined as `is_correct=True` and low `answer_min_logprob`/`min_logprob`.
- In this run, protected cases are `NOT_AVAILABLE because no measured prediction/logprob file was provided`.

## 7. Next Experiment Candidates

1. cryptarithm carry-focused +100（推奨度：★★★★★）
   - 理由: carry mistakes are solver-checkable and can be trained with a narrow template.
   - baselineとの差分: add only 100 carry-focused cryptarithm examples.
   - 変更対象: next-phase training data only; no adapter structure change.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_carry --n 100`
   - 評価方法: rerun Phase3 validation and compare cryptarithm carry accuracy plus protected-case retention.
   - 採用条件: carry subset improves without reducing stable-correct categories.
   - rollback方法: remove the +100 carry data file and restore previous adapter checkpoint.
   - 失敗リスク: overfitting to addition wording or disturbing already-correct alphametics.

2. cryptarithm leading-zero +100（推奨度：★★★★☆）
   - 理由: leading-zero constraints are explicit, solver-checkable, and easy to isolate.
   - baselineとの差分: add only leading-zero constraint examples.
   - 変更対象: next-phase training data only.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_leading_zero --n 100`
   - 評価方法: leading-zero failure-type count and final accuracy.
   - 採用条件: fewer leading-zero errors with no parse regression.
   - rollback方法: remove leading-zero data and checkpoint.
   - 失敗リスク: model may over-mention leading-zero constraints when absent.

3. cryptarithm mapping-conflict +100（推奨度：★★★★☆）
   - 理由: distinct-digit mapping errors are common in alphametic reasoning and solver-verifiable.
   - baselineとの差分: add only mapping-conflict examples.
   - 変更対象: next-phase training data only.
   - 実行コマンド: `python make_sft_data.py --type cryptarithm_mapping_conflict --n 100`
   - 評価方法: mapping-conflict failure count and protected cryptarithm retention.
   - 採用条件: improvement on conflict cases without arithmetic/carry regression.
   - rollback方法: remove mapping-conflict data and checkpoint.
   - 失敗リスク: longer search traces may hurt concise final-answer parsing.

4. bit XOR/base conversion +100（推奨度：★★★☆☆）
   - 理由: bitwise XOR plus base conversion is solver-checkable and complements cryptarithm.
   - baselineとの差分: add only XOR/base conversion examples.
   - 変更対象: next-phase data only.
   - 実行コマンド: `python make_sft_data.py --type bit_xor_base --n 100`
   - 評価方法: bit_manipulation accuracy and parse success.
   - 採用条件: measurable bit subset gain with no cryptarithm regression.
   - rollback方法: remove XOR/base data and checkpoint.
   - 失敗リスク: base conversion artifacts can leak into numeral-only problems.

5. numeral conversion +100（推奨度：★★★☆☆）
   - 理由: deterministic place-value tasks are low-risk and solver-checkable.
   - baselineとの差分: add only numeral-conversion examples.
   - 変更対象: next-phase data only.
   - 実行コマンド: `python make_sft_data.py --type numeral_conversion --n 100`
   - 評価方法: numeral_conversion accuracy and protected-case retention.
   - 採用条件: improves numeral subset without answer-format degradation.
   - rollback方法: remove numeral data and checkpoint.
   - 失敗リスク: easy examples may not transfer to harder reasoning tasks.

6. answer format / final parse template only（推奨度：★★★☆☆）
   - 理由: targets parse failures only and avoids changing reasoning content.
   - baselineとの差分: prompt/template post-processing only; no SFT data change.
   - 変更対象: final answer extraction template.
   - 実行コマンド: `python run_validation.py --template final_parse_v2`
   - 評価方法: parse success and no accuracy loss on stable-correct cases.
   - 採用条件: parse_success improves and accuracy does not drop.
   - rollback方法: revert template file.
   - 失敗リスク: format changes can alter exact answer matching.

7. min logprob lower-tail replay candidate（推奨度：★★★☆☆）
   - 理由: replays low-confidence correct and wrong examples to protect fragile wins and target uncertain misses.
   - baselineとの差分: select only lower-tail logprob examples; no architecture change.
   - 変更対象: next-phase data sampling list.
   - 実行コマンド: `python select_lower_tail.py --input phase3_analysis/min_logprob_summary.csv`
   - 評価方法: retention of correct-low-confidence cases plus wrong-low-confidence recovery.
   - 採用条件: protects fragile correct examples while improving selected misses.
   - rollback方法: remove lower-tail replay list and checkpoint.
   - 失敗リスク: requires reliable logprobs; without logprobs this is HOLD.

## 8. ADOPT / HOLD / REJECT

### ADOPT

- Adopt only categories that have enough local failures, solver verification, synthetic generation, and no protected-case regression in Phase3 outputs.

### HOLD

- Hold all confidence/logprob claims when `min_logprob_summary.csv` lacks logprob values.
- Hold categories with low count, ambiguous classification, or validation skew.

### REJECT

- Reject adapter-rank/target-module/dtype changes, Golden adapter edits, public-LB-only feedback loops, and multi-variable experiments in the next phase.

## Failure Type Summary

| category | failure_type | n | share |
|---|---:|---:|---:|
| ALL | NOT_AVAILABLE | 0 |  |
