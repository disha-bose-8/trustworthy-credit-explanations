# python -m src.export_for_powerbi   (builds applicants.csv and shap_long.csv)
from pathlib import Path
import numpy as np, pandas as pd
from src.data_prep import READABLE, FEATURES

OUT = Path("outputs/powerbi"); OUT.mkdir(parents=True, exist_ok=True)
clean = pd.read_csv("data/processed/clean.csv")
pred = pd.read_csv("data/processed/test_predictions.csv")
shap = pd.read_csv("data/processed/shap_values_test.csv")

# 1. Fix the 300 applicants once, so the LLM step uses exactly these people
sel = pred[pred["flagged"] == 1].sample(300, random_state=42)["SK_ID_CURR"]
sel.to_frame("applicant_id").to_csv(OUT / "selected_300.csv", index=False)

# 2. applicants.csv: the 300 plus a bigger random sample (for Key Influencers)
pool = pred[~pred["SK_ID_CURR"].isin(sel)]
extra = pool.sample(min(9700, len(pool)), random_state=42)
keep = pd.concat([pred[pred["SK_ID_CURR"].isin(sel)], extra])
app = keep.merge(clean.drop(columns=["TARGET"]), on="SK_ID_CURR")
app["in_300"] = app["SK_ID_CURR"].isin(sel).astype(int)
app = app.rename(columns=READABLE).rename(columns={"SK_ID_CURR": "applicant_id"})
app.to_csv(OUT / "applicants.csv", index=False)

# 3. shap_long.csv: top 10 features per selected applicant
s = shap[shap["SK_ID_CURR"].isin(sel)].melt(id_vars="SK_ID_CURR", var_name="feature",
                                             value_name="shap_value")
vals = clean[clean["SK_ID_CURR"].isin(sel)].set_index("SK_ID_CURR")[FEATURES]
s["feature_value"] = [vals.at[i, f] for i, f in zip(s["SK_ID_CURR"], s["feature"])]
s["shap_rank"] = (s.assign(a=s["shap_value"].abs())
                    .groupby("SK_ID_CURR")["a"].rank(ascending=False, method="first").astype(int))
s = s[s["shap_rank"] <= 10].copy()
s["direction"] = np.where(s["shap_value"] > 0, "increases risk", "decreases risk")
s["feature_readable"] = s["feature"].map(READABLE)
s = s.rename(columns={"SK_ID_CURR": "applicant_id"})
s[["applicant_id", "feature_readable", "feature_value", "shap_value",
   "shap_rank", "direction"]].to_csv(OUT / "shap_long.csv", index=False)
print("applicants.csv", app.shape, "| shap_long.csv", s.shape)