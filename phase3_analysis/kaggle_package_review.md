# Kaggle Phase3 Package Review

## Package status

Reviewed `phase3_package.zip` from the Kaggle first-pass run. The package contains 16 files, including `phase3_analysis/category_map.csv`, `phase3_analysis/validation_set_labeled.csv`, `phase3_analysis/phase3_recommendation.md`, `phase3_validation_split.csv`, and `phase3_validation_split.manifest.json`.

## Split validation

The split manifest is internally consistent:

| field | value |
|---|---:|
| source train rows | 9500 |
| validation rows | 1900 |
| remainder rows | 7600 |
| fraction | 0.2 |
| seed | 42 |
| preserved columns | `id`, `prompt`, `answer` |

This confirms that the first-pass deterministic CV split succeeded.

## Measurement status

The first pass intentionally supplied no Golden prediction/logprob artifacts:

| field | value |
|---|---:|
| prediction rows | 0 |
| logprob rows | 0 |
| Golden accuracy | `NOT_AVAILABLE` |
| min logprob analysis | `NOT_AVAILABLE` |

This is expected because the current operating priority is category-map/CV-design analysis without modifying Golden Baseline logging.

## Category distribution from uploaded package

The uploaded package's generated `category_map.csv` contains 1900 validation rows. Counts from that package were:

| category | rows | share |
|---|---:|---:|
| bit_manipulation | 815 | 42.9% |
| numeral_conversion | 640 | 33.7% |
| cipher | 298 | 15.7% |
| equation | 147 | 7.7% |

Important audit finding: the original classifier over-counted `bit_manipulation` because formula expressions such as `t^2` were interpreted as caret/XOR. It also under-specified Roman numeral rows as generic `base_n_conversion`.

## Refined classifier preview

After reviewing the package, the classifier was updated and re-run locally against `phase3_validation_split.csv`. A second refinement removed symbolic-equation false positives for `&`, `|`, `<<`, and `>>`. The refined preview distribution is:

| category | rows | share |
|---|---:|---:|
| numeral_conversion | 640 | 33.7% |
| bit_manipulation | 335 | 17.6% |
| arithmetic | 317 | 16.7% |
| equation | 310 | 16.3% |
| cipher | 298 | 15.7% |

Refined subcategory preview:

| category | subcategory | rows |
|---|---|---:|
| bit_manipulation | xor | 335 |
| numeral_conversion | roman_numeral | 330 |
| arithmetic | formula_evaluation | 317 |
| numeral_conversion | base_n_conversion | 310 |
| equation | equation_solving | 310 |
| cipher | cipher | 298 |

Manual review flags in the corrected preview: 317 rows, corresponding to the arithmetic/formula-evaluation group. These should be reviewed before being used as a targeted next-phase training category.

## Code updates made from this review

1. Bare `^` is no longer treated as bitwise XOR when it appears as exponent notation such as `t^2`.
2. Symbolic operators `^`, `&`, `|`, `<<`, and `>>` are treated as bit operators only when the prompt has bit/binary/operator context, reducing symbolic-equation false positives.
3. Roman numeral examples such as `76 -> LXXVI` are detected before generic base conversion.
4. Physics/formula prompts such as falling-distance examples are classified as `arithmetic/formula_evaluation` and marked for manual review.

## Recommended next action

Re-run the updated `phase3_offline_analysis.ipynb` or updated `phase3_make_recommendation.py` in Kaggle against the same split to regenerate the package with corrected category labels. Golden prediction/logprob metrics can remain blank for this pass.

Expected first-pass success criteria after rerun:

- `validation_rows = 1900`;
- `prediction_rows = 0` and `logprob_rows = 0` remain acceptable;
- category counts should be close to the refined preview above;
- `phase3_validation_split.manifest.json` should still report `n_total = 9500`, `n_validation = 1900`, `seed = 42`.
