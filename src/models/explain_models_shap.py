import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

# =========================
# Config
# =========================
DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")
MODELS_ROOT = Path("artifacts/models")
TARGET_COL = "binary_label"

SHAP_SAMPLE_SIZE = 2000
TOP_K = 5
RANDOM_STATE = 42


# =========================
# Helpers
# =========================
def get_latest_run_dir(models_root: Path) -> Path:
    run_dirs = [d for d in models_root.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError("No model run directories found")

    return sorted(run_dirs)[-1]


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

    print(f"\nTop {TOP_K} SHAP features for model: {model_name}")
    for feat, val in importance.items():
        print(f"  {feat:<30} {val:.5f}")

    return importance


# =========================
# Main
# =========================
def main():
    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])

    X_sample = X.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X)),
        random_state=RANDOM_STATE
    )

    print(f"Using SHAP sample: {len(X_sample)} rows")

    run_dir = get_latest_run_dir(MODELS_ROOT)
    print(f"Using latest model run: {run_dir.name}")

    model_paths = list(run_dir.glob("*.joblib"))
    if not model_paths:
        raise RuntimeError(f"No .joblib models found in {run_dir}")

    for model_path in model_paths:
        model = joblib.load(model_path)
        explain_model(model_path.stem, model, X_sample)


if __name__ == "__main__":
    main()
