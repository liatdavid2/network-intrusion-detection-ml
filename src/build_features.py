import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold


# =========================
# Paths
# =========================
RAW_DATA_PATH = Path("data/raw/UNSW_Flow.parquet")
OUTPUT_DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")

TARGET_COL = "binary_label"


# =========================
# Columns to exclude
# =========================
NON_FEATURE_COLS = [
    "source_ip",
    "destination_ip",
    "protocol",
    "state",
    "service",
    "attack_label",
    "binary_label",
]

IDENTIFIER_COLS = [
    "src_ip",
    "dst_ip",
    "flow_id",
    "session_id",
    "timestamp",
    "start_time",
    "end_time",
]


# =========================
# Preprocessing helpers
# =========================
def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in IDENTIFIER_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def variance_filter(df: pd.DataFrame, threshold: float = 0.0):
    selector = VarianceThreshold(threshold=threshold)
    values = selector.fit_transform(df)

    selected_features = df.columns[selector.get_support()]
    df_selected = pd.DataFrame(values, columns=selected_features, index=df.index)

    return df_selected, list(selected_features)


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    df_filtered = df.drop(columns=to_drop)
    return df_filtered, to_drop


# =========================
# Main pipeline
# =========================
def build_features():
    print(f"Loading data from {RAW_DATA_PATH}")
    df = pd.read_parquet(RAW_DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    # Log non-numeric columns (debug / transparency)
    non_numeric_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print("Non-numeric columns detected:")
    print(non_numeric_cols)

    # Separate target
    y = df[TARGET_COL]

    # Drop target + known non-features
    X = df.drop(
        columns=[c for c in NON_FEATURE_COLS if c in df.columns],
        errors="ignore"
    )

    # Drop identifier columns
    X = drop_identifier_columns(X)

    # Keep numeric features only (critical)
    X = X.select_dtypes(include=[np.number])

    print(f"Numeric feature count before filtering: {X.shape[1]}")

    # Handle missing values
    X = handle_missing_values(X)

    # Variance filter
    X, variance_features = variance_filter(X, threshold=0.0)
    print(f"Features after variance filter: {X.shape[1]}")

    # Correlation filter
    X, dropped_corr_features = correlation_filter(X, threshold=0.95)
    print(f"Features after correlation filter: {X.shape[1]}")

    # Reattach target
    X[TARGET_COL] = y.values

    # Save output
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUTPUT_DATA_PATH, index=False)

    print(f"Saved processed features to {OUTPUT_DATA_PATH}")
    print(f"Final number of features (excluding target): {X.shape[1] - 1}")

    print("Final feature list:")
    for col in X.columns:
        if col != TARGET_COL:
            print(col)

    print("Target distribution:")
    print(y.value_counts(normalize=True))

    return list(X.columns)


if __name__ == "__main__":
    build_features()
