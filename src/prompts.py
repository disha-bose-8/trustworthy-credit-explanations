# The 3 prompts. They are IDENTICAL except for the evidence block.
import numpy as np, pandas as pd

CONDITIONS = ["none", "global", "applicant-level"]

def _fmt(v):
    if pd.isna(v):
        return "not available"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)

def _evidence(condition, global_rank, truth):
    if condition == "none":
        return ""
    if condition == "global":
        lines = [f"{k}. {f}" for k, f in enumerate(global_rank, 1)]
        return ("EVIDENCE\nAcross all applicants, the model relies most on these features "
                "(most important first):\n" + "\n".join(lines) + "\n\n")
    top = truth.reindex(truth.abs().sort_values(ascending=False).index)[:10]
    lines = [f"{k}. {f}: {'increases' if s > 0 else 'decreases'} risk (SHAP {s:+.2f})"
             for k, (f, s) in enumerate(top.items(), 1)]
    return ("EVIDENCE\nThe model's own SHAP analysis for THIS applicant "
            "(largest influence first):\n" + "\n".join(lines) + "\n\n")

def build_prompt(condition, values, features, global_rank, truth):
    applicant = "\n".join(f"- {f}: {_fmt(values[f])}" for f in features)
    allowed = "\n".join(f"- {f}" for f in features)
    return (
        "You are explaining a loan-risk model's prediction to a loan officer.\n\n"
        f"APPLICANT\n{applicant}\n\n"
        "The model flagged this applicant as having a high risk of defaulting on the loan.\n\n"
        f"{_evidence(condition, global_rank, truth)}"
        "TASK\n"
        "Name the 3 features that most influenced the model's risk prediction for this applicant. "
        "For each, say whether it increases risk or decreases risk. Then write a 2-3 sentence "
        "plain-English explanation for the loan officer.\n"
        "Choose feature names only from this list, copied exactly:\n"
        f"{allowed}\n\n"
        "Reply with JSON only, no other text, in exactly this shape:\n"
        '{"drivers": [{"feature": "<name>", "direction": "<increases risk or decreases risk>"}, '
        '{"feature": "<name>", "direction": "<increases risk or decreases risk>"}, '
        '{"feature": "<name>", "direction": "<increases risk or decreases risk>"}], '
        '"narrative": "<2-3 sentences>"}'
    )