import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

# =========================
# Config
# =========================
DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")
RUN_DIR = Path("artifacts/models/20260202_121045")  # ← לשנות לפי הריצה
TARGET_COL = "binary_label"

SHAP_SAMPLE_SIZE = 2000
TOP_K = 5
RANDOM_STATE = 42


# =========================
# Helper
# =========================
def explain_model(model_name, model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Binary classification → class 1 (attack)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = (
        pd.Series(mean_abs_shap, index=X.columns)
        .sort_values(ascending=False)
        .head(TOP_K)
    )

    print(f"\nTop {TOP_K} SHAP features for {model_name}:")
    for feat, val in importance.items():
        print(f"  {feat:<25} {val:.4f}")

    return importance


# =========================
# Main
# =========================
def main():
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])

    # sample - small and fast
    X_sample = X.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X)),
        random_state=RANDOM_STATE
    )

    print(f"Using SHAP sample: {len(X_sample)} rows")

    for model_path in RUN_DIR.glob("*.joblib"):
        model_name = model_path.stem
        model = joblib.load(model_path)

        explain_model(model_name, model, X_sample)


if __name__ == "__main__":
    main()
