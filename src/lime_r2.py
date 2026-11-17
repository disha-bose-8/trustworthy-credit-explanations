"""Run LIME once on the 300 study applicants and record each local R².

Run from the repo root:  python -m src.lime_r2
Writes outputs/tables/lime_local_r2.csv (one row per applicant).
R² = how well LIME's simple local model matches the LightGBM model near that applicant.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.lime_explain import SEED, load_inputs, encode, make_predict, LimeTabularExplainer
from src.data_prep import CATEGORICAL, FEATURES

OUT = Path("outputs/tables/lime_local_r2.csv")


def main():
    model, apps, X = load_inputs()
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

    rows = apps.index[apps["in_300"] == 1]
    out = []
    for i in rows:
        exp = explainer.explain_instance(Z[i], predict, num_features=len(FEATURES),
                                         num_samples=5000, labels=(1,))
        out.append({"applicant_id": apps.loc[i, "applicant_id"],
                    "local_r2": round(exp.score, 4)})

    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Saved {len(df)} applicants to {OUT}")
    print(f"Local R²: mean {df.local_r2.mean():.3f}, median {df.local_r2.median():.3f}, "
          f"min {df.local_r2.min():.3f}, max {df.local_r2.max():.3f}")
    for t in [0.5, 0.7, 0.9]:
        print(f"  applicants with R² >= {t}: {(df.local_r2 >= t).sum()} ({(df.local_r2 >= t).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
