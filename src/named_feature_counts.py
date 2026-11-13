"""How often each condition names a feature in its top 3, versus the SHAP top 3.
Run from the repo root:  python named_feature_counts.py
Reads outputs/llm_cache/*.json, outputs/powerbi/selected_300.csv, outputs/powerbi/shap_long.csv."""
import collections, json, re
import pandas as pd

CONDS = ["none", "global", "applicant-level"]
ids = pd.read_csv("outputs/powerbi/selected_300.csv")["applicant_id"].tolist()
named = {c: collections.Counter() for c in CONDS}
for i in ids:
    for c in CONDS:
        text = json.load(open(f"outputs/llm_cache/{i}_{c}.json"))["text"]
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        for x in d["drivers"][:3]:           # same truncation as generate_explanations.parse
            named[c][x["feature"]] += 1

s = pd.read_csv("outputs/powerbi/shap_long.csv")
true = s[s["shap_rank"] <= 3]["feature_readable"].value_counts()

feats = ["External credit score 3", "External credit score 2", "External credit score 1",
         "Years in current job", "Loan-to-income ratio", "Credit bureau enquiries (last year)"]
rows = [[f, *[named[c][f] for c in CONDS], int(true.get(f, 0))] for f in feats]
print(pd.DataFrame(rows, columns=["feature", *CONDS, "SHAP_top3"]).to_string(index=False))
print("applicants:", len(ids))
