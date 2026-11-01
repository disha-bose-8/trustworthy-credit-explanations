"""Compute SHAP values (the ground truth) for every unseen test applicant."""
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_prep import CATEGORICAL, FEATURES, READABLE
from src.train_model import FIG, MODEL_PATH, PRED_PATH, load_config, load_data, split

SHAP_PATH = Path("data/processed/shap_values_test.csv")
TABLES = Path("outputs/tables")

# shap prints a harmless notice about its LightGBM output format; the code below handles both formats.
warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

TOP_K = 8  # features shown in the single-applicant chart


def fmt(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{value:,.2f}"
    return str(value)


def main() -> None:
    cfg = load_config()
    model = joblib.load(MODEL_PATH)
    _, _, test = split(load_data(), cfg["seed"], cfg["test_size"])
    X = test[FEATURES]

    # 1. SHAP values: one number per applicant per feature
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # some shap versions return [class 0, class 1]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # some return (applicants, features, classes)
        sv = sv[:, :, 1]
    base = np.ravel(explainer.expected_value)[-1]

    # 2. Sanity check: baseline + sum of an applicant's SHAP values must equal the model's raw output
    raw = model.predict(X, raw_score=True)
    err = np.abs(base + sv.sum(axis=1) - raw).max()
    print(f"Additivity check (should be ~0): max difference = {err:.2e}")

    shap_df = pd.DataFrame(sv, columns=FEATURES)
    shap_df.insert(0, "SK_ID_CURR", test["SK_ID_CURR"].values)
    shap_df.to_csv(SHAP_PATH, index=False)

    # 3. Global importance: average size of each feature's push across all test applicants
    mean_abs = np.abs(sv).mean(axis=0)
    glob = pd.DataFrame({"feature": FEATURES, "readable": [READABLE[f] for f in FEATURES], "mean_abs_shap": mean_abs})
    glob["share_pct"] = glob["mean_abs_shap"] / glob["mean_abs_shap"].sum() * 100
    glob = glob.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    glob.insert(0, "rank", glob.index + 1)
    TABLES.mkdir(parents=True, exist_ok=True)
    glob.to_csv(TABLES / "global_importance.csv", index=False)
    print("\nTop 10 features overall (global ranking):")
    print(glob.head(10)[["rank", "readable", "share_pct"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    FIG.mkdir(parents=True, exist_ok=True)

    top = glob.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["readable"], top["mean_abs_shap"], color="#028090")
    ax.set_xlabel("Average |SHAP value| (average push on the model's output)")
    ax.set_title("Global feature importance (all test applicants)")
    plt.tight_layout()
    plt.savefig(FIG / "08_global_shap_importance.png", dpi=150)
    plt.close()

    # 4. Beeswarm: every dot is one applicant. Text columns are turned into codes just for colouring.
    Xp = X.copy()
    for c in CATEGORICAL:
        Xp[c] = Xp[c].cat.codes
    Xp.columns = [READABLE[c] for c in Xp.columns]
    shap.summary_plot(sv, Xp, max_display=12, show=False)
    plt.title("SHAP summary (red = high feature value, blue = low)")
    plt.tight_layout()
    plt.savefig(FIG / "09_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. One example flagged applicant: the chart your LLM explanations will be checked against
    pred = pd.read_csv(PRED_PATH)
    flagged = pred[pred["flagged"] == 1].sample(1, random_state=cfg["seed"]).iloc[0]
    pos = int(np.where(test["SK_ID_CURR"].values == flagged["SK_ID_CURR"])[0][0])
    row, vals = sv[pos], X.iloc[pos]
    order = np.argsort(-np.abs(row))[:TOP_K][::-1]
    labels = [f"{READABLE[FEATURES[i]]} = {fmt(vals.iloc[i])}" for i in order]
    colors = ["#d9534f" if row[i] > 0 else "#4c9fa8" for i in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, row[order], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (red pushes toward default, teal pushes toward repaying)")
    ax.set_title(f"Applicant {int(flagged['SK_ID_CURR'])}: predicted default probability {flagged['pred_prob']:.1%}")
    plt.tight_layout()
    plt.savefig(FIG / "10_example_applicant_shap.png", dpi=150)
    plt.close()

    print(f"\nExample applicant {int(flagged['SK_ID_CURR'])} (predicted default probability {flagged['pred_prob']:.1%}, "
          f"actually {'defaulted' if flagged['TARGET'] == 1 else 'repaid'}):")
    for i in order[::-1][:5]:
        direction = "pushes toward DEFAULT" if row[i] > 0 else "pushes toward REPAYING"
        print(f"  {READABLE[FEATURES[i]]} = {fmt(vals.iloc[i])}: {row[i]:+.3f} ({direction})")


if __name__ == "__main__":
    main()