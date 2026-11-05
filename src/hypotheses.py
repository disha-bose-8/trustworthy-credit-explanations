# python -m src.hypotheses
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

TABLES = Path("outputs/tables"); TABLES.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/processed/clean.csv")
rows = []

def groups(col):
    return df.loc[df.TARGET == 1, col].dropna(), df.loc[df.TARGET == 0, col].dropna()

def welch(col, label):
    a, b = groups(col)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    rows.append([label, "Welch t-test", t, p, (a.mean()-b.mean())/sp, "Cohen's d", 0.05])

def mannwhitney(col, label):
    a, b = groups(col)
    u, p = stats.mannwhitneyu(a, b)
    rows.append([label, "Mann-Whitney U", u, p, 2*u/(len(a)*len(b)) - 1, "rank-biserial r", 0.05])

def chisq(col, label):
    ct = pd.crosstab(df[col], df.TARGET)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    rows.append([label, "Chi-square", chi2, p, v, "Cramer's V", 0.05])

welch("EXT_SOURCE_2", "H1 External credit score 2 vs default")
mannwhitney("CREDIT_TO_INCOME", "H1b Loan-to-income ratio vs default")
chisq("NAME_EDUCATION_TYPE", "H2 Education level vs default")

# H3 runs only after the LLM step has written explanations_long.csv
EXPL = Path("outputs/powerbi/explanations_long.csv")
if EXPL.exists():
    ex = pd.read_csv(EXPL)
    w = ex.pivot(index="applicant_id", columns="condition", values="top3_overlap")
    for a, b in [("none", "global"), ("global", "applicant-level"), ("none", "applicant-level")]:
        d = (w[b] - w[a]).dropna()
        try:
            stat, p = stats.wilcoxon(d)
        except ValueError:          # all differences are zero
            stat, p = np.nan, 1.0
        rows.append([f"H3 {a} vs {b} (top-3 overlap)", "Wilcoxon signed-rank",
                     stat, p, d.mean(), "mean gain in overlap", 0.05/3])

out = pd.DataFrame(rows, columns=["hypothesis", "test", "statistic", "p_value",
                                  "effect_size", "effect_type", "alpha"])
out["significant"] = out["p_value"] < out["alpha"]
out.to_csv(TABLES / "hypothesis_results.csv", index=False)
print(out.to_string(index=False))