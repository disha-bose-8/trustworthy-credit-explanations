# Trustworthy Credit Explanations

Do LLM-written explanations of credit-risk decisions actually reflect what the model used? This project trains a credit default model, computes SHAP values as ground truth, asks an LLM to explain flagged applicants under three evidence conditions, and scores how faithful each explanation is.

## Pipeline

1. **Model**: LightGBM default-risk classifier on the [Home Credit](https://www.kaggle.com/c/home-credit-default-risk) `application_train.csv` data (test AUC ≈ 0.75).
2. **Ground truth**: SHAP values for every unseen test applicant.
3. **Explanations**: an LLM (Groq) explains 300 flagged applicants under three prompt conditions, identical except for the evidence block:
   - `none`: no evidence
   - `global`: global feature importance ranking
   - `applicant-level`: the applicant's own SHAP values
4. **Faithfulness metrics**: top-3 feature overlap with SHAP, sign (direction) accuracy, and count of invented features.
5. **Statistics and dashboard**: hypothesis tests and a Power BI dashboard.

## Repo structure

```
src/        pipeline code (data prep, EDA, training, SHAP, prompts, LLM client, metrics, hypotheses)
configs/    config.yaml (seed, split, number of applicants, LLM settings)
outputs/    figures, tables, model metrics, Power BI CSVs, cached LLM responses
*.pbix      Power BI dashboard (+ exported .html version)
```

## Setup

```bash
pip install -r requirements.txt
pip install groq                      # or swap the client in src/llm_client.py
export GROQ_API_KEY="your-key"
```

Download `application_train.csv` from Kaggle into `data/raw/`.

## Run

```bash
python -m src.eda
python -m src.train_model
python -m src.compute_shap
python -m src.export_for_powerbi
python -m src.generate_explanations --limit 10   # pilot; omit --limit for all 300
python -m src.hypotheses
python -m src.named_feature_counts
```

LLM responses are cached in `outputs/llm_cache/`, so reruns don't repeat API calls.

## Notes

- Only `src/llm_client.py` depends on the LLM provider.
- `src/mock_explanations.py` produces fake data for dashboard layout testing only.
