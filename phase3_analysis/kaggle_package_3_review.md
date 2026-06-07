# Kaggle Phase3 Package 3 Review

## Reviewed package

Reviewed `phase3_package (3).zip` from branch `codex/conduct-phase-3-analysis-for-nemotron-00nzc6`.

## Status

This package is a successful first-pass category-map/CV-design run. It contains the deterministic validation split, manifest, populated `category_map.csv`, and populated `validation_set_labeled.csv`.

## Split validation

| field | value |
|---|---:|
| source train rows | 9500 |
| validation rows | 1900 |
| remainder rows | 7600 |
| fraction | 0.2 |
| seed | 42 |
| preserved columns | `id`, `prompt`, `answer` |

## Uploaded package category distribution

The uploaded package produced:

| category | rows | share |
|---|---:|---:|
| numeral_conversion | 640 | 33.7% |
| bit_manipulation | 418 | 22.0% |
| arithmetic | 317 | 16.7% |
| cipher | 298 | 15.7% |
| equation | 227 | 11.9% |

Subcategory distribution from the uploaded package:

| category | subcategory | rows |
|---|---|---:|
| bit_manipulation | xor | 335 |
| numeral_conversion | roman_numeral | 330 |
| arithmetic | formula_evaluation | 317 |
| numeral_conversion | base_n_conversion | 310 |
| cipher | cipher | 298 |
| equation | equation_solving | 227 |
| bit_manipulation | and | 45 |
| bit_manipulation | or | 26 |
| bit_manipulation | unknown | 7 |
| bit_manipulation | shift_left | 4 |
| bit_manipulation | shift_right | 1 |

## Remaining issue found

The package still uses the pre-refinement symbolic-operator logic. Rows with `&`, `|`, `<<`, or `>>` in symbolic equation-rule prompts are still counted as bit manipulation. These should be equation-style rule-induction/equation rows unless the prompt has explicit bit/binary/operator context.

## Refined local rerun result

Using the current repo classifier on the same `phase3_validation_split.csv` gives:

| category | rows | share |
|---|---:|---:|
| numeral_conversion | 640 | 33.7% |
| bit_manipulation | 335 | 17.6% |
| arithmetic | 317 | 16.7% |
| equation | 310 | 16.3% |
| cipher | 298 | 15.7% |

The generated README/recommendation docs were also fixed so category-map-only runs report `Validation rows analyzed: 1900` instead of using the prediction-row count of zero.

## Measurement status

`prediction_rows = 0` and `logprob_rows = 0` remain expected for first-pass analysis. Golden accuracy and logprob summaries remain `NOT_AVAILABLE` because no Golden prediction/logprob artifacts were supplied.

## Recommended next action

Rerun the latest notebook/script after ensuring the GitHub notebook contains `symbolic_bit_tokens`. Success target:

- `validation_rows = 1900`;
- README status `CATEGORY_MAP_ONLY_NO_PREDICTION_LOGS`;
- `prediction_rows = 0` and `logprob_rows = 0` acceptable;
- category counts close to numeral_conversion=640, bit_manipulation=335, arithmetic=317, equation=310, cipher=298.
