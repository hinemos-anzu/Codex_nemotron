# Kaggle Phase3 Package 2 Review

## Reviewed package

Reviewed `phase3_package (2).zip` from branch `codex/conduct-phase-3-analysis-for-nemotron-00nzc6`.

## Status

This package is a successful first-pass category-map/CV-design run. It contains `phase3_validation_split.csv`, `phase3_validation_split.manifest.json`, populated `category_map.csv`, and populated `validation_set_labeled.csv`.

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

The uploaded package used the first classifier fix and produced:

| category | rows | share |
|---|---:|---:|
| numeral_conversion | 640 | 33.7% |
| bit_manipulation | 418 | 22.0% |
| arithmetic | 317 | 16.7% |
| cipher | 298 | 15.7% |
| equation | 227 | 11.9% |

## Additional audit finding

The package still had 7 `bit_manipulation/unknown` rows. Inspection showed these were symbolic equation-rule prompts containing `<<` or `>>`, not actual bit-shift tasks. Similar symbolic-equation rows had previously appeared with `&` and `|`.

## Refined classifier result

After the additional refinement, symbolic operators `^`, `&`, `|`, `<<`, and `>>` require bit/binary/operator context before being treated as bit manipulation. Re-running locally on the same `phase3_validation_split.csv` gives:

| category | rows | share |
|---|---:|---:|
| numeral_conversion | 640 | 33.7% |
| bit_manipulation | 335 | 17.6% |
| arithmetic | 317 | 16.7% |
| equation | 310 | 16.3% |
| cipher | 298 | 15.7% |

Subcategory distribution:

| category | subcategory | rows |
|---|---|---:|
| bit_manipulation | xor | 335 |
| numeral_conversion | roman_numeral | 330 |
| arithmetic | formula_evaluation | 317 |
| numeral_conversion | base_n_conversion | 310 |
| equation | equation_solving | 310 |
| cipher | cipher | 298 |

Manual review flags remain 317 rows, corresponding to the arithmetic/formula-evaluation group.

## Measurement status

`prediction_rows = 0` and `logprob_rows = 0` remain expected because this pass intentionally avoids Golden Baseline logging. Golden accuracy and logprob summaries remain `NOT_AVAILABLE`.

## Recommended next action

Rerun the updated notebook/script once more to regenerate `phase3_package.zip` with the refined symbolic-operator classifier. The successful first-pass target remains:

- `validation_rows = 1900`;
- `prediction_rows = 0`;
- `logprob_rows = 0`;
- category counts close to numeral_conversion=640, bit_manipulation=335, arithmetic=317, equation=310, cipher=298.
