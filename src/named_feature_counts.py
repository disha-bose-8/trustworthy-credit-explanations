"""How often each condition names a feature in its top 3, versus the SHAP top 3, for all 25 features.
Run from the repo root:  python named_feature_counts.py
Reads outputs/llm_cache/*.json, outputs/powerbi/selected_300.csv, outputs/powerbi/shap_long.csv,
outputs/tables/global_importance.csv. Writes named_feature_table.tex (the paper's table)."""
import collections, json, re
import pandas as pd

CONDS = ["none", "global", "applicant-level"]

def normalize_name(name):
    """Same as src/metrics.py: case-folded, whitespace-collapsed."""
    return " ".join(str(name).casefold().split())

g = pd.read_csv("outputs/tables/global_importance.csv").sort_values("rank")
ALLOWED = g["readable"].tolist()                       # the 25 features, most important first
LOOKUP = {normalize_name(f): f for f in ALLOWED}

ids = pd.read_csv("outputs/powerbi/selected_300.csv")["applicant_id"].tolist()
named = {c: collections.Counter() for c in CONDS}
for i in ids:
    for c in CONDS:
        text = json.load(open(f"outputs/llm_cache/{i}_{c}.json"))["text"]
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        for x in d["drivers"][:3]:                     # same truncation as generate_explanations.parse
            canon = LOOKUP.get(normalize_name(x["feature"]))
            assert canon is not None, f"name outside the 25 allowed features: {x['feature']!r}"
            named[c][canon] += 1

s = pd.read_csv("outputs/powerbi/shap_long.csv")
true = s[s["shap_rank"] <= 3]["feature_readable"].value_counts()

rows = [[f, *[named[c][f] for c in CONDS], int(true.get(f, 0))] for f in ALLOWED]
totals = [sum(r[k] for r in rows) for k in range(1, 5)]
assert totals == [3 * len(ids)] * 4, totals            # every column must sum to 900
print(pd.DataFrame(rows, columns=["feature", *CONDS, "SHAP_top3"]).to_string(index=False))
print("column totals:", totals)
print("applicants:", len(ids))

# LaTeX tabular for the paper (\input inside the table environment).
def esc(t): return t.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
EOL = " \\\\"
L = [r"\begin{tabular}{r>{\raggedright\arraybackslash}p{3.7cm}cccc}", r"    \toprule",
     r"    \textbf{\#} & \textbf{Feature} & \textbf{None} & \textbf{Global} & \textbf{Appl.} & \textbf{SHAP}" + EOL,
     r"    \midrule"]
L += [f"    {k} & {esc(r[0])} & " + " & ".join(str(v) for v in r[1:]) + EOL for k, r in enumerate(rows, 1)]
L += [r"    \midrule", "    & Total & " + " & ".join(str(t) for t in totals) + EOL, r"    \bottomrule", r"\end{tabular}"]
open("named_feature_table.tex", "w").write("\n".join(L) + "\n")
