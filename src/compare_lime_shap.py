
"""Compare LIME's top-3 features with SHAP's top-3 (the ground truth) for the 300 applicants.

Run from the repo root, after src.lime_explain:  python -m src.compare_lime_shap
Writes:
  outputs/tables/lime_vs_shap_per_applicant.csv  (one row per applicant)
  outputs/tables/lime_vs_shap_per_feature.csv    (how often each feature is named by each method)
  outputs/figures/lime_vs_shap.png               (agreement chart)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIME_PATH = Path("outputs/tables/lime_top3.csv")
SHAP_PATH = Path("outputs/powerbi/shap_long.csv")
OUT_TABLES = Path("outputs/tables")
OUT_FIG = Path("outputs/figures")


def load():
    lime = pd.read_csv(LIME_PATH)
    shap = pd.read_csv(SHAP_PATH)
    shap_top3 = shap[shap["shap_rank"] <= 3][["applicant_id", "feature_readable", "direction"]]
    lime_top3 = lime[["applicant_id", "feature", "direction"]].rename(columns={"direction": "lime_direction"})
    return lime_top3, shap_top3


def per_applicant(lime_top3, shap_top3):
    """One row per applicant: which features each method named, and how many agree."""
    rows = []
    for aid, lg in lime_top3.groupby("applicant_id"):
        sg = shap_top3[shap_top3.applicant_id == aid]
        lime_feats = set(lg["feature"])
        shap_feats = set(sg["feature_readable"])
        both = lime_feats & shap_feats

        # Direction agreement on the features both methods name in their top 3.
        sdir = dict(zip(sg["feature_readable"], sg["direction"]))
        ldir = dict(zip(lg["feature"], lg["lime_direction"]))
        dir_match = [ldir[f] == sdir[f] for f in both]

        rows.append({
            "applicant_id": aid,
            "lime_top3": "; ".join(sorted(lime_feats)),
            "shap_top3": "; ".join(sorted(shap_feats)),
            "agreed_features": len(both),
            "overlap": len(both) / 3,
            "direction_agreement_on_shared": round(np.mean(dir_match), 3) if dir_match else np.nan,
        })
    return pd.DataFrame(rows)


def per_feature(lime_top3, shap_top3):
    """For each feature: how often SHAP names it, how often LIME names it, and how often both do."""
    lime_sets = lime_top3.groupby("applicant_id")["feature"].apply(set)
    shap_sets = shap_top3.groupby("applicant_id")["feature_readable"].apply(set)
    feats = sorted(set(lime_top3["feature"]) | set(shap_top3["feature_readable"]))
    rows = []
    for f in feats:
        in_shap = sum(f in s for s in shap_sets)
        in_lime = sum(f in s for s in lime_sets)
        in_both = sum(f in lime_sets[a] and f in shap_sets[a] for a in lime_sets.index if a in shap_sets.index)
        rows.append({
            "feature": f,
            "named_by_shap_top3": in_shap,
            "named_by_lime_top3": in_lime,
            "named_by_both": in_both,
            "lime_only": in_lime - in_both,
            "shap_only": in_shap - in_both,
        })
    return pd.DataFrame(rows).sort_values("named_by_shap_top3", ascending=False)


def print_summary(app):
    n = len(app)
    print("=" * 60)
    print(f"LIME vs SHAP top-3 agreement, {n} flagged applicants")
    print("=" * 60)
    print(f"Mean top-3 overlap: {app['overlap'].mean():.3f}   (random guessing ~0.12)")
    print("\nHow many of the 3 features match, per applicant:")
    for k in [3, 2, 1, 0]:
        c = (app["agreed_features"] == k).sum()
        print(f"  {k} of 3 match: {c:4d} applicants ({c / n * 100:5.1f}%)")
    d = app["direction_agreement_on_shared"].dropna()
    print(f"\nDirection agreement on shared features (both say 'increases' or 'decreases'): {d.mean() * 100:.1f}%")
    print(f"  applicants with at least one shared feature: {len(d)}")


def plot(app, feat):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts = app["agreed_features"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    axes[0].bar([f"{k} of 3" for k in counts.index], counts.values,
                color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, f"{v / len(app) * 100:.0f}%", ha="center")
    axes[0].set_ylabel("Applicants")
    axes[0].set_title("How many of LIME's top 3 match SHAP's top 3")

    top = feat.head(10).iloc[::-1]
    y = np.arange(len(top))
    axes[1].barh(y + 0.27, top["named_by_shap_top3"], height=0.25, color="#028090", label="SHAP top 3")
    axes[1].barh(y, top["named_by_lime_top3"], height=0.25, color="#f0ad4e", label="LIME top 3")
    axes[1].barh(y - 0.27, top["named_by_both"], height=0.25, color="#5cb85c", label="Both")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(top["feature"], fontsize=8)
    axes[1].set_xlabel("Applicants naming this feature in their top 3")
    axes[1].set_title("Which features each method names")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = OUT_FIG / "lime_vs_shap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved chart: {path}")


def main():
    lime_top3, shap_top3 = load()
    app = per_applicant(lime_top3, shap_top3)
    feat = per_feature(lime_top3, shap_top3)

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    app.to_csv(OUT_TABLES / "lime_vs_shap_per_applicant.csv", index=False)
    feat.to_csv(OUT_TABLES / "lime_vs_shap_per_feature.csv", index=False)

    print_summary(app)
    print("\nFeatures named by each method (top 5 by SHAP):")
    print(feat.head(5).to_string(index=False))
    plot(app, feat)
    print(f"\nSaved tables to {OUT_TABLES}/lime_vs_shap_per_*.csv")


if __name__ == "__main__":
    main()
