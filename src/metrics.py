# The match score. Compares what the LLM said with the SHAP ground truth.
'''import pandas as pd

def score(drivers, truth, allowed):
    """drivers: [(feature, direction), ...] from the LLM (3 items).
    truth: Series, readable feature name -> this applicant's SHAP value.
    allowed: list of the real feature names."""
    true_top3 = set(truth.abs().sort_values(ascending=False).index[:3])
    named = {f for f, _ in drivers}
    valid = [(f, d) for f, d in drivers if f in allowed]
    invented = len(drivers) - len(valid)
    overlap = len(named & true_top3) / 3
    if valid:
        right = sum(("increase" in d.lower()) == (truth[f] > 0) for f, d in valid)
        sign = right / len(valid)
    else:
        sign = 0.0
    return {"top3_overlap": round(overlap, 4), "sign_accuracy": round(sign, 4),
            "invented_count": invented}

if __name__ == "__main__":     # python -m src.metrics  (hand-check on 3 toy cases)
    truth = pd.Series({"A": 3.0, "B": -2.0, "C": 1.0, "D": 0.1})
    allowed = ["A", "B", "C", "D"]
    cases = [
        ("perfect",      [("A","increases risk"),("B","decreases risk"),("C","increases risk")], (1.0, 1.0, 0)),
        ("one hit, one invented", [("A","increases risk"),("D","increases risk"),("Z","increases risk")], (1/3, 1.0, 1)),
        ("right features, all signs wrong", [("B","increases risk"),("D","decreases risk"),("C","decreases risk")], (2/3, 0.0, 0)),
    ]
    for name, drivers, (o, s, n) in cases:
        r = score(drivers, truth, allowed)
        assert abs(r["top3_overlap"] - o) < 1e-3 and abs(r["sign_accuracy"] - s) < 1e-3 and r["invented_count"] == n, (name, r)
        print("OK ", name, r)'''

# The match score. Compares what the LLM said with the SHAP ground truth.
import pandas as pd


def normalize_name(name):
    """Make feature-name matching case-insensitive and ignore extra spaces."""
    return " ".join(str(name).casefold().split())


def score(drivers, truth, allowed):
    """Compare LLM drivers with this applicant's actual SHAP values."""

    # Map normalized names back to the official feature names.
    allowed_lookup = {normalize_name(f): f for f in allowed}

    # Convert LLM feature names to their official names where possible.
    valid = []
    for feature, direction in drivers:
        canonical = allowed_lookup.get(normalize_name(feature))
        if canonical is not None:
            valid.append((canonical, direction))

    # Count names that genuinely don't match any allowed feature.
    invented = len(drivers) - len(valid)

    # Compare the matched features with the actual top 3 SHAP features.
    true_top3 = set(
        truth.abs().sort_values(ascending=False).index[:3]
    )
    named = {feature for feature, _ in valid}
    overlap = len(named & true_top3) / 3

    # Check whether the LLM got the direction right.
    if valid:
        right = sum(
            ("increase" in direction.lower()) == (truth[feature] > 0)
            for feature, direction in valid
        )
        sign = right / len(valid)
    else:
        sign = 0.0

    return {
        "top3_overlap": round(overlap, 4),
        "sign_accuracy": round(sign, 4),
        "invented_count": invented
    }


if __name__ == "__main__":
    truth = pd.Series({"A": 3.0, "B": -2.0, "C": 1.0, "D": 0.1})
    allowed = ["A", "B", "C", "D"]

    cases = [
        ("perfect", [
            ("A", "increases risk"),
            ("B", "decreases risk"),
            ("C", "increases risk")
        ], (1.0, 1.0, 0)),

        ("one hit, one invented", [
            ("A", "increases risk"),
            ("D", "increases risk"),
            ("Z", "increases risk")
        ], (1/3, 1.0, 1)),

        ("right features, all signs wrong", [
            ("B", "increases risk"),
            ("D", "decreases risk"),
            ("C", "decreases risk")
        ], (2/3, 0.0, 0)),
    ]

    for name, drivers, (o, s, n) in cases:
        r = score(drivers, truth, allowed)
        assert abs(r["top3_overlap"] - o) < 1e-3
        assert abs(r["sign_accuracy"] - s) < 1e-3
        assert r["invented_count"] == n
        print("OK", name, r)