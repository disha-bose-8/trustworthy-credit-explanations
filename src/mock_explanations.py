# python -m src.mock_explanations   LAYOUT TESTING ONLY, fake numbers
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("outputs/powerbi")
ids = pd.read_csv(OUT / "selected_300.csv")["applicant_id"]
rng = np.random.default_rng(0)
rows = []
for i in ids:
    for order, cond, p in [(1, "none", .3), (2, "global", .55), (3, "applicant-level", .9)]:
        ov = rng.binomial(3, p) / 3
        rows.append([i, cond, order, "MOCK DATA - not a real LLM output", ov,
                     rng.choice([0, .5, 1], p=[.2, .3, .5]), 0])
pd.DataFrame(rows, columns=["applicant_id", "condition", "condition_order", "narrative",
             "top3_overlap", "sign_accuracy", "invented_count"]).to_csv(OUT / "explanations_long.csv", index=False)
print("mock written:", len(rows), "rows")