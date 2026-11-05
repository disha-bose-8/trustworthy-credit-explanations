# python -m src.generate_explanations --limit 10    (pilot on 10 applicants)
# python -m src.generate_explanations               (all 300)
import argparse, json, re
from pathlib import Path
import pandas as pd
from src.data_prep import READABLE
from src.prompts import build_prompt, CONDITIONS
from src.metrics import score
from src.llm_client import call_llm

OUT = Path("outputs/powerbi")
CACHE = Path("outputs/llm_cache"); CACHE.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

FEATS = list(READABLE.values())
app = pd.read_csv(OUT / "applicants.csv")
app = app[app["in_300"] == 1].set_index("applicant_id")
ids = pd.read_csv(OUT / "selected_300.csv")["applicant_id"].tolist()
if args.limit:
    ids = ids[:args.limit]
shap = pd.read_csv("data/processed/shap_values_test.csv").set_index("SK_ID_CURR").rename(columns=READABLE)
global_rank = shap.abs().mean().sort_values(ascending=False).index.tolist()[:10]

def parse(text):
    d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    return [(x["feature"], x["direction"]) for x in d["drivers"]][:3], d["narrative"]

rows, failed = [], 0
for n, i in enumerate(ids, 1):
    truth, values = shap.loc[i], app.loc[i, FEATS]
    for order, cond in enumerate(CONDITIONS, 1):
        f = CACHE / f"{i}_{cond}.json"          # cached, so reruns cost nothing
        if f.exists():
            text = json.loads(f.read_text())["text"]
        else:
            text = call_llm(build_prompt(cond, values, FEATS, global_rank, truth))
            f.write_text(json.dumps({"text": text}))
        try:
            drivers, narrative = parse(text)
        except Exception:
            failed += 1
            print("could not parse", i, cond, "-> delete", f, "to retry")
            continue
        rows.append({"applicant_id": i, "condition": cond, "condition_order": order,
                     "narrative": narrative, **score(drivers, truth, FEATS)})
    print(f"{n}/{len(ids)} done")

df = pd.DataFrame(rows)
df.to_csv(OUT / "explanations_long.csv", index=False)       # overwrites the MOCK file
print("\nrows:", len(df), "| parse failures:", failed)
print(df.groupby("condition")[["top3_overlap", "sign_accuracy", "invented_count"]].mean().round(3))