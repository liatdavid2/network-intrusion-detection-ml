"""
Risk Framing Script
-------------------
Loads the latest trained model, scores network flows,
and produces deterministic, human-readable risk explanations
based on raw feature thresholds and model score.

No SHAP. No model introspection.
Explanations are rule-based and production-oriented.

Usage:
    python src/risk/run_risk_framing.py
"""

from pathlib import Path
import json
import joblib
import pandas as pd


# =========================
# Paths & Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "final_model"
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "UNSW_Flow_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "risk_framing_output.json"


# =========================
# Risk Rules (Domain-Based)
# =========================

RISK_RULES = [
    {
        "feature": "dur",
        "condition": lambda v: v > 10,
        "reason": "Unusually long connection duration",
    },
    {
        "feature": "sbytes",
        "condition": lambda v: v > 1_000_000,
        "reason": "Large amount of outbound data",
    },
    {
        "feature": "dbytes",
        "condition": lambda v: v > 1_000_000,
        "reason": "Large amount of inbound data",
    },
    {
        "feature": "ct_dst_sport_ltm",
        "condition": lambda v: v > 50,
        "reason": "High fan-out to destination ports",
    },
    {
        "feature": "ct_srv_dst",
        "condition": lambda v: v > 100,
        "reason": "High number of connections to the same destination",
    },
]


# =========================
# Helpers
# =========================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No model runs found in {base_dir}")
    return max(run_dirs, key=lambda d: d.name)


def assign_risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


def explain_row(row: pd.Series) -> list[str]:
    reasons = []
    for rule in RISK_RULES:
        value = row.get(rule["feature"])
        if value is None:
            continue
        try:
            if rule["condition"](value):
                reasons.append(rule["reason"])
        except Exception:
            continue
    return reasons


# =========================
# Main
# =========================

def main():
    print("Loading model artifacts...")

    run_dir = get_latest_run_dir(FINAL_MODEL_ROOT)
    print(f"Using model run: {run_dir.name}")

    model = joblib.load(run_dir / "final_model.joblib")
    feature_names = load_json(run_dir / "feature_names.json")
    decision_policy = load_json(run_dir / "decision_policy.json")
    threshold = decision_policy["selected_threshold"]

    print("Loading input data...")
    df = pd.read_parquet(INPUT_FILE).reset_index(drop=True)

    # Align features exactly as during training
    X = df[feature_names].copy().fillna(0)

    print("Running model inference...")
    scores = model.predict_proba(X)[:, 1]

    print("Framing risk explanations...")
    results = []

    for i, row in df.iterrows():
        score = float(scores[i])
        risk_level = assign_risk_level(score, threshold)

        reasons = explain_row(row)

        # Always include model-based reason if score is meaningful
        if score >= threshold:
            reasons.insert(
                0,
                f"Model confidence above threshold ({score:.2f} ≥ {threshold:.2f})",
            )

        results.append(
            {
                "flow_index": i,
                "risk_score": round(score, 4),
                "risk_level": risk_level,
                "reasons": reasons,
            }
        )

    print(f"Saving output to {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
