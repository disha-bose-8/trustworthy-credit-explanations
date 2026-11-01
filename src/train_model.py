"""Train the LightGBM risk model (Model 1) and save its predictions on unseen applicants."""
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src.data_prep import CATEGORICAL, CLEAN_PATH, FEATURES

MODEL_PATH = Path("outputs/lgbm_model.joblib")
PRED_PATH = Path("data/processed/test_predictions.csv")
METRICS_PATH = Path("outputs/model_metrics.json")
FIG = Path("outputs/figures")

# Newer LightGBM versions print a harmless notice about the `eval_set` argument name.
warnings.filterwarnings("ignore", message=".*eval_set.*")

# Team decision: applicants in the riskiest X% of the test set count as "flagged" (rejected).
FLAG_TOP_PCT = 10


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    for c in CATEGORICAL:  # LightGBM reads text columns natively once they are "category"
        df[c] = df[c].astype("category")
    return df


def split(df: pd.DataFrame, seed: int, test_size: float):
    """Same seed -> same split every time, so every script sees the same test applicants."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["TARGET"], random_state=seed)
    train, valid = train_test_split(train_val, test_size=0.1, stratify=train_val["TARGET"], random_state=seed)
    return train, valid, test


def main() -> None:
    cfg = load_config()
    df = load_data()
    train, valid, test = split(df, cfg["seed"], cfg["test_size"])
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=cfg["seed"], verbose=-1,
    )
    model.fit(
        train[FEATURES], train["TARGET"],
        eval_set=[(valid[FEATURES], valid["TARGET"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],  # stop when validation stops improving
    )

    p = model.predict_proba(test[FEATURES])[:, 1]  # predicted probability of default
    auc = roc_auc_score(test["TARGET"], p)

    out = test[["SK_ID_CURR", "TARGET"]].copy()
    out["pred_prob"] = p
    threshold = out["pred_prob"].quantile(1 - FLAG_TOP_PCT / 100)
    out["flagged"] = (out["pred_prob"] >= threshold).astype(int)
    flagged_rate = out.loc[out["flagged"] == 1, "TARGET"].mean()
    overall_rate = out["TARGET"].mean()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    out.to_csv(PRED_PATH, index=False)
    METRICS_PATH.write_text(json.dumps({
        "test_auc": round(float(auc), 4), "trees_used": int(model.best_iteration_),
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "flag_top_pct": FLAG_TOP_PCT, "flag_threshold_prob": round(float(threshold), 4),
        "default_rate_overall_test": round(float(overall_rate), 4),
        "default_rate_among_flagged": round(float(flagged_rate), 4),
    }, indent=2))

    fpr, tpr, _ = roc_curve(test["TARGET"], p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#028090", label=f"LightGBM (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Coin flip (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve on unseen applicants")
    ax.legend()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / "07_roc_curve.png", dpi=150)
    plt.close()

    print(f"Test AUC: {auc:.4f}  (trees used: {model.best_iteration_})")
    print(f"Riskiest {FLAG_TOP_PCT}% flagged (prob >= {threshold:.3f}): "
          f"{flagged_rate * 100:.1f}% of them actually defaulted vs {overall_rate * 100:.1f}% overall")
    print(f"Saved: {MODEL_PATH}, {PRED_PATH}, {METRICS_PATH}")


if __name__ == "__main__":
    main()