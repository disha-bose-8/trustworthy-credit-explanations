"""EDA (Unit 1): saves figures to outputs/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files, don't open pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_prep import CATEGORICAL, FEATURES, READABLE, clean, load_raw, select

FIG = Path("outputs/figures")
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("saved", FIG / name)


raw = load_raw()
df = select(clean(raw))

# 1. Class balance: how many applicants default?
rates = df["TARGET"].value_counts(normalize=True).sort_index() * 100
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Repaid (0)", "Defaulted (1)"], rates.values, color=["#4c9fa8", "#d9534f"])
for i, v in enumerate(rates.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
ax.set_ylabel("% of applicants")
ax.set_title(f"Class balance: {rates[1]:.1f}% of applicants default")
save("01_class_balance.png")

# 2. Missing values: which columns are mostly empty?
miss = (df[FEATURES].isna().mean() * 100).sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([READABLE[c] for c in miss.index][::-1], miss.values[::-1], color="#4c9fa8")
ax.set_xlabel("% missing")
ax.set_title("Most incomplete model features")
save("02_missing_values.png")

# 3. The DAYS_EMPLOYED placeholder (looked at BEFORE cleaning)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(raw["DAYS_EMPLOYED"], bins=60, color="#d9534f")
ax.set_yscale("log")
ax.set_xlabel("DAYS_EMPLOYED (raw value)")
ax.set_ylabel("Applicants (log scale)")
ax.set_title("Raw DAYS_EMPLOYED: spike at 365243 is a placeholder")
save("03_days_employed_anomaly.png")

# 4. External credit scores, split by outcome
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
    for target, label, color in [(0, "Repaid", "#4c9fa8"), (1, "Defaulted", "#d9534f")]:
        sns.kdeplot(df.loc[df["TARGET"] == target, col].dropna(), ax=ax,
                    label=label, color=color, fill=True, alpha=0.3)
    ax.set_title(READABLE[col])
    ax.set_xlabel("score")
    ax.legend()
save("04_external_scores_by_outcome.png")

# 5. Correlation of each numeric feature with TARGET
num = df.drop(columns=["SK_ID_CURR"]).select_dtypes("number")
corr = num.corr()["TARGET"].drop("TARGET")
top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:12].sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([READABLE.get(c, c) for c in top.index], top.values,
        color=["#d9534f" if v > 0 else "#4c9fa8" for v in top.values])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Correlation with default (TARGET)")
ax.set_title("Features most correlated with default")
save("05_correlation_with_target.png")

# 6. Default rate by category (ignore tiny groups, they are noisy)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col in zip(axes, ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"]):
    g = df.groupby(col)["TARGET"].agg(["mean", "size"])
    g = g[g["size"] >= 500].sort_values("mean")
    ax.barh(g.index, g["mean"] * 100, color="#4c9fa8")
    ax.set_xlabel("Default rate (%)")
    ax.set_title(READABLE[col])
save("06_default_rate_by_category.png")

print(f"\n{df.shape[0]:,} applicants, {len(FEATURES)} model features, "
      f"default rate {df['TARGET'].mean() * 100:.2f}%")