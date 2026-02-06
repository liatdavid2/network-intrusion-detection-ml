import json
import joblib
import pandas as pd

# paths
MODEL_PATH = "artifacts/models/20260206_180557/HistGradientBoosting.joblib"
FEATURES_PATH = "artifacts/models/20260206_180557/feature_names.json"
INPUT_PATH = "src/inference/examples_api_samples/flow_borderline.json"

# load
model = joblib.load(MODEL_PATH)
feature_names = json.load(open(FEATURES_PATH))
flow = json.load(open(INPUT_PATH))

# build X
X = pd.DataFrame(
    [[flow.get(f, 0) for f in feature_names]],
    columns=feature_names
)

# predict
prob = model.predict_proba(X)[0, 1]
formatted = f"{prob:.7f}"
print("risk_score:", formatted)

