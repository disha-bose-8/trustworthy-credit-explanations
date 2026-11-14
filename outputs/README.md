# Outputs

Everything the pipeline produces. Run commands from the repo root. `data/` is git-ignored, so regenerate it by following the main README.

## Contents

| Path | What it is | Produced by |
|---|---|---|
| `model_metrics.json` | Test AUC, split sizes, flagging threshold and default rates | `src.train_model` |
| `lgbm_model.joblib` | Trained LightGBM classifier | `src.train_model` |
| `figures/` | 10 numbered PNGs (index below) | `src.eda`, `src.train_model`, `src.compute_shap` |
| `tables/` | Global SHAP importance and hypothesis test results | `src.compute_shap`, `src.hypotheses` |
| `powerbi/` | CSVs that feed the Power BI dashboard | `src.export_for_powerbi`, `src.generate_explanations` |
| `llm_cache/` | Cached raw LLM replies, 900 JSON files | `src.generate_explanations` |

## `model_metrics.json`

| Key | Meaning |
|---|---|
| `test_auc` | ROC AUC on the held-out test set (0.7528) |
| `trees_used` | Number of trees kept after early stopping (277) |
| `n_train`, `n_valid`, `n_test` | Rows in each split (221,407 / 24,601 / 61,503) |
| `flag_top_pct` | Top X% of test applicants by predicted risk are "flagged" (10) |
| `flag_threshold_prob` | Predicted probability cutoff that corresponds to that flag (0.1773) |
| `default_rate_overall_test` | Actual default rate across the test set (8.07%) |
| `default_rate_among_flagged` | Actual default rate among flagged applicants (27.26%) |

## `figures/`

| File | Shows |
|---|---|
| `01_class_balance.png` | Default vs non-default counts |
| `02_missing_values.png` | Missing values by feature |
| `03_days_employed_anomaly.png` | The anomalous placeholder values in the employment-days field |
| `04_external_scores_by_outcome.png` | External credit score distributions by default outcome |
| `05_correlation_with_target.png` | Feature correlations with default |
| `06_default_rate_by_category.png` | Default rate across categorical groups |
| `07_roc_curve.png` | ROC curve of the default model |
| `08_global_shap_importance.png` | Mean absolute SHAP value per feature |
| `09_shap_beeswarm.png` | SHAP value distribution per feature across test applicants |
| `10_example_applicant_shap.png` | SHAP breakdown for one example applicant |

## `tables/`

**`global_importance.csv`**: one row per model feature, ranked by mean absolute SHAP value.

| Column | Meaning |
|---|---|
| `rank` | 1 = most important |
| `feature` | Raw column name (e.g. `EXT_SOURCE_3`) |
| `readable` | Human-readable name used in prompts and the dashboard |
| `mean_abs_shap` | Mean absolute SHAP value |
| `share_pct` | Share of total importance, in % |

**`hypothesis_results.csv`**: one row per statistical test.

| Column | Meaning |
|---|---|
| `hypothesis` | Label (H1, H1b, H2 are about the data; H3 compares evidence conditions) |
| `test` | Test used (Welch t-test, Mann-Whitney U, chi-square, Wilcoxon signed-rank) |
| `statistic`, `p_value` | Test statistic and p-value |
| `effect_size`, `effect_type` | Effect size and its type (Cohen's d, rank-biserial r, Cramér's V, mean gain in overlap) |
| `alpha` | Significance threshold. H3 rows use 0.05/3 (Bonferroni, 3 comparisons) |
| `significant` | `p_value < alpha` |

## `powerbi/`

**`selected_300.csv`**: the `applicant_id`s of the 300 applicants explained by the LLM, sampled from flagged test applicants with seed 42.

**`applicants.csv`**: 10,000 test applicants (the 300 selected plus 9,700 random others, used for Power BI's Key Influencers visual). Columns include `applicant_id`, `TARGET` (1 = defaulted), `pred_prob` (model-predicted default probability), `flagged` (1 = in the top-10% risk group), the readable feature columns, and `in_300` (1 = one of the 300 explained applicants).

**`shap_long.csv`**: top 10 SHAP features for each of the 300 applicants (3,000 rows).

| Column | Meaning |
|---|---|
| `applicant_id` | Applicant |
| `feature_readable` | Feature name |
| `feature_value` | The applicant's value for that feature |
| `shap_value` | SHAP contribution (positive pushes risk up) |
| `shap_rank` | 1 = largest absolute SHAP value for this applicant |
| `direction` | `increases risk` or `decreases risk` |

**`explanations_long.csv`**: one row per applicant per condition (900 rows). Main file for the faithfulness analysis.

| Column | Meaning |
|---|---|
| `applicant_id` | Applicant |
| `condition` | Evidence given to the LLM: `none`, `global` or `applicant-level` |
| `condition_order` | 1, 2, 3 for none, global, applicant-level (for sorting in Power BI) |
| `narrative` | The LLM's written explanation |
| `top3_overlap` | Fraction of the applicant's true SHAP top-3 features that the LLM named. One of 0, 1/3, 2/3, 1 |
| `sign_accuracy` | Of the named features, the fraction whose stated direction (increases/decreases risk) matches the sign of that feature's SHAP value. 0 if no named feature is valid |
| `invented_count` | Number of named features (out of 3) that don't match any real model feature (case-insensitive) |

Scoring logic lives in `src/metrics.py`; run `python -m src.metrics` for a hand-checked sanity test.

## `llm_cache/`

One file per applicant per condition, named `{applicant_id}_{condition}.json` (e.g. `104106_applicant-level.json`). Each holds `{"text": "<raw LLM reply>"}`, where the reply is JSON containing three `drivers` (feature + direction) and a `narrative`.

- Reruns of `src.generate_explanations` skip any applicant/condition that already has a cache file, so no API calls are repeated.
- File names don't encode the model or prompt. If you change the LLM or the prompts, delete the cache first to regenerate.
- `src.named_feature_counts` reads this folder to count how often each feature is named per condition.