"""Cross-check the SHAP ground truth with LIME for the 300 flagged applicants.

Run from the repo root:  python -m src.lime_explain
Writes outputs/tables/lime_top3.csv (one row per applicant, LIME's top 3 features).
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.data_prep import CATEGORICAL, FEATURES, READABLE

warnings.filterwarnings("ignore")
SEED = 42
NUM_SAMPLES = 5000     # LIME's default perturbation count
OUT = Path("outputs/tables/lime_top3.csv")


def load_inputs():
    model = joblib.load("outputs/lgbm_model.joblib")
    apps = pd.read_csv("outputs/powerbi/applicants.csv")
    X = apps[[READABLE[f] for f in FEATURES]].copy()
    X.columns = FEATURES
    return model, apps, X


def encode(X, codes):
    """LIME works on numbers, so categories become integer codes."""
    Z = X.copy()
    for c in CATEGORICAL:
        Z[c] = pd.Categorical(Z[c], categories=codes[c]).codes
    return Z.astype(float).values


def make_predict(model, codes):
    """Turn LIME's numeric rows back into the model's input and return [P(repay), P(default)]."""
    def predict(arr):
        d = pd.DataFrame(arr, columns=FEATURES)
        for c in CATEGORICAL:
            idx = d[c].round().astype(int).clip(0, len(codes[c]) - 1)
            d[c] = pd.Categorical.from_codes(idx, categories=codes[c])
        p = model.predict_proba(d)[:, 1]
        return np.column_stack([1 - p, p])
    return predict


def main(n_applicants=None):
    model, apps, X = load_inputs()

    # LIME cannot handle missing values, so fill numeric gaps with the median
    # for the explainer only. LightGBM itself still sees the original data.
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    X[numeric] = X[numeric].astype(float).fillna(X[numeric].median())
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    codes = {c: X[c].cat.categories for c in CATEGORICAL}

    Z = encode(X, codes)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]
    explainer = LimeTabularExplainer(
        Z, feature_names=FEATURES, class_names=["repay", "default"],
        categorical_features=cat_idx,
        categorical_names={i: list(codes[FEATURES[i]]) for i in cat_idx},
        mode="classification", random_state=SEED,
    )
    predict = make_predict(model, codes)

    rows = apps.index[apps["in_300"] == 1]  # the 300 applicants used in the study
    if n_applicants:
        rows = rows[:n_applicants]

    out = []
    for i in rows:
        exp = explainer.explain_instance(Z[i], predict, num_features=len(FEATURES),
                                         num_samples=NUM_SAMPLES, labels=(1,))
        # as_map gives (feature index, weight) pairs; keep the 3 largest by |weight|.
        top = sorted(exp.as_map()[1], key=lambda t: -abs(t[1]))[:3]
        for rank, (idx, weight) in enumerate(top, start=1):
            out.append({
                "applicant_id": apps.loc[i, "applicant_id"],
                "rank": rank,
                "feature": READABLE[FEATURES[idx]],
                "lime_weight": round(weight, 4),
                "direction": "increases risk" if weight > 0 else "decreases risk",
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lime = pd.DataFrame(out)
    lime.to_csv(OUT, index=False)
    print(f"Wrote {len(lime)} rows ({lime.applicant_id.nunique()} applicants) to {OUT}")

    # Agreement with the SHAP top 3 (the ground truth used in the study).
    shap = pd.read_csv("outputs/powerbi/shap_long.csv")
    shap_top3 = shap[shap.shap_rank <= 3].groupby("applicant_id")["feature_readable"].apply(set)
    lime_top3 = lime.groupby("applicant_id")["feature"].apply(set)
    overlap = [len(lime_top3[a] & shap_top3[a]) / 3 for a in lime_top3.index if a in shap_top3.index]
    print(f"Mean top-3 overlap, LIME vs SHAP: {np.mean(overlap):.3f} over {len(overlap)} applicants")


if __name__ == "__main__":
    main()