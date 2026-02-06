# Network Intrusion Detection – Machine Learning

This project implements a machine learning pipeline for detecting network intrusions based on structured network flow data.  
It includes data loading, feature engineering, model training, and evaluation.

## Setup

Clone the repository and enter the project directory:
```bash
git clone https://github.com/liatdavid2/network-intrusion-detection-ml.git
cd network-intrusion-detection-ml
````

Create and activate a virtual environment:

Windows (Git Bash):

```bash
py -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project structure

```
network-intrusion-detection-ml/          # Root of the project repository

├── .venv/                               # Local Python virtual environment (development only)

├── artifacts/                           # All generated artifacts (models, metrics, configs)
│   ├── final_model/                     # Final, production-ready model runs
│   │   └── 20260204_153203/              # Timestamped model run (single immutable experiment)
│   │       ├── evaluation/              # Evaluation outputs on the test set
│   │       │   ├── roc_curve.png         # ROC curve of the final model
│   │       │   ├── pr_curve.png          # Precision-Recall curve
│   │       │   └── test_metrics.json     # Final test metrics (Precision, Recall, F1, etc.)
│   │       │
│   │       ├── decision_policy.json      # Decision thresholds and risk policy configuration
│   │       ├── feature_names.json        # Ordered list of features used during training
│   │       ├── final_model.joblib        # Serialized trained model used for inference
│   │       ├── run_config.json           # Training configuration (seed, sampling, parameters)
│   │       └── threshold_analysis.csv    # Threshold vs. metric analysis used to select cutoff
│   │
│   └── models/                          # Baseline or intermediate models (optional / exploratory)

├── data/                                # Dataset storage
│   ├── raw/                             # Original UNSW-NB15 data (never modified)
│   └── processed/                       # Cleaned and feature-engineered datasets

├── notebooks/                           # Exploratory notebooks (EDA, analysis, experiments)

├── src/                                 # All executable source code
│   ├── build_features.py                # Feature engineering pipeline (raw → processed)
│   │
│   ├── models/                          # Model training logic
│   │   ├── train_baseline_models.py     # Training of baseline models for comparison
│   │   └── train_final_model.py         # Training of the selected final model
│   │
│   ├── evaluation/                      # Model evaluation logic
│   │   └── evaluate_final_model.py      # Evaluation on held-out test set
│   │
│   └── inference/                       # Inference and API simulation layer
│       ├── predict.py                   # Production-style CLI for single & batch inference
│       ├── make_batch_flows.py          # Utility to create mixed attack/benign batch inputs
│       │
│       └── examples_api_samples/        # Example inputs for API / CLI simulation
│           ├── flow_low_risk.json        # Single benign flow example
│           ├── flow_high_risk.json       # Single attack flow example
│           └── batch_flows.parquet       # Mixed batch of benign + attack flows

├── .gitignore                           # Git ignore rules (data, artifacts, venv, etc.)
├── README.md                            # Project documentation
└── requirements.txt                    # Python dependencies

```

## Step 1 – EDA
Exploratory data analysis is performed in the following notebook:

`notebooks/01_dataset_familiarization_unsw_nb15.ipynb`

## Step 2 – Run feature engineering

Run feature engineering:
```bash
python src/build_features.py
```
Expected output:

```text
[1/6] Load data ............... OK (2,059,415 rows)
[2/6] Target detected ......... binary_label (4.84% attacks)
[3/6] Drop non-numeric ........ 6 columns removed
[4/6] Numeric features ........ 42
[5/6] Feature selection ....... 42 → 37
[6/6] Save features ........... data/processed/UNSW_Flow_features.parquet

Class distribution:
  normal : 95.16%
  attack :  4.84%
```

The processed dataset is saved to:

```text
data/processed/UNSW_Flow_features.parquet
```
---

## Step 3 – Training & Baseline Models

In this step, we train and compare several baseline machine-learning models on the engineered feature set.

### Run training

```bash
python src/models/train_baseline_models.py
```

### Training pipeline

The script performs the following steps:

1. **Run directory creation**
   Each run is stored under `artifacts/models/<timestamp>/` for full reproducibility.

2. **Feature loading**
   Loads the full engineered feature table.
   Total samples: **2,059,415 rows**

3. **Stratified subsampling**
   To reduce training time while preserving class distribution,
   a **20% stratified subsample** is used: **411,883 rows**

4. **Train / Test split**
   Stratified split with an **80 / 20** ratio.

5. **Model training & evaluation**
   Multiple baseline models are trained and evaluated using:

   * ROC-AUC
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

---

### Models evaluated

* RandomForest
* ExtraTrees
* GradientBoosting
* HistGradientBoosting
* XGBoost

---

### Results

#### Per-model metrics

```
RandomForest
ROC-AUC=0.9992 | Precision=0.9489 | Recall=0.9084 | F1=0.9282
Confusion Matrix:
[[78196   195]
 [  365  3621]]

ExtraTrees
ROC-AUC=0.9991 | Precision=0.9329 | Recall=0.9109 | F1=0.9218
Confusion Matrix:
[[78130   261]
 [  355  3631]]

GradientBoosting
ROC-AUC=0.9990 | Precision=0.9130 | Recall=0.9109 | F1=0.9120
Confusion Matrix:
[[78045   346]
 [  355  3631]]

HistGradientBoosting
ROC-AUC=0.9993 | Precision=0.9427 | Recall=0.9250 | F1=0.9338
Confusion Matrix:
[[78167   224]
 [  299  3687]]

XGBoost
ROC-AUC=0.9993 | Precision=0.7826 | Recall=0.9952 | F1=0.8762
Confusion Matrix:
[[77289  1102]
 [   19  3967]]
```

---

### Baseline comparison (sorted by ROC-AUC)

| Model                | ROC-AUC | Precision | Recall |     F1 |
| -------------------- | ------: | --------: | -----: | -----: |
| HistGradientBoosting |  0.9993 |    0.9427 | 0.9250 | 0.9338 |
| XGBoost              |  0.9993 |    0.7826 | 0.9952 | 0.8762 |
| RandomForest         |  0.9992 |    0.9489 | 0.9084 | 0.9282 |
| ExtraTrees           |  0.9991 |    0.9329 | 0.9109 | 0.9218 |
| GradientBoosting     |  0.9990 |    0.9130 | 0.9109 | 0.9120 |

---
## Step 4 – Model Improvement & Decision Threshold Selection

In this step, the focus shifts from model comparison to **operational decision quality**.
Rather than optimizing ROC-AUC further, we introduce an explicit **decision threshold** to control false positives and make the model production-ready.

### Run training

```bash
python src/models/train_final_model.py
```

---

### What happens in this step

1. **Final model selection**
   Based on Step 3 baseline results, the model with the best overall balance between Precision and Recall is selected:

   * **HistGradientBoostingClassifier**

2. **Score-based evaluation**
   The model is evaluated using predicted probabilities (`predict_proba`) rather than hard labels.
   This enables separation between **risk estimation** and **decision logic**.

3. **Automatic threshold discovery**
   Instead of using a fixed threshold (e.g., 0.5), the script:

   * Evaluates multiple thresholds in the range `[0.01 – 0.99]`
   * Computes Precision, Recall, F1, and false-positive statistics for each
   * Selects the threshold that **maximizes F1 score**

   The threshold value is **empirically derived from the data**, not chosen arbitrarily.

4. **Decision policy creation**
   The selected threshold becomes a first-class component of the system and is stored alongside the model for future inference.

---

### Example output

```
[0/4] Run directory .......... artifacts/final_model/20260204_153203
[1/4] Load features .......... OK (2,059,415 rows)
[2/4] Train/Test split ...... OK (80/20, stratified)
[3/4] Training final model ... HistGradientBoosting
[4/4] ROC-AUC (scores only) . 0.9993
[DONE] Selected threshold ...... 0.480 (Precision=0.937, Recall=0.933)
```

**Interpretation:**

* The threshold value (0.480) is **not a confidence level**
* It is the cutoff at which alerts achieve ~94% precision while maintaining high recall
* This provides strong detection capability without overwhelming the SOC with false positives

---

### Artifacts produced

Each run generates a fully versioned, production-ready package:

* `final_model.joblib` – trained model
* `decision_policy.json` – selected threshold and selection rationale
* `threshold_analysis.csv` – metrics across all tested thresholds
* `feature_names.json` – feature ordering for safe inference
* `run_config.json` – training configuration for reproducibility

---

### Why this matters

Separating **risk scoring** from **decision thresholds** allows:

* Controlled false-positive rates
* Threshold changes without retraining
* Alignment with SOC capacity and risk tolerance

This step transforms a high-performing model into a **deployable detection system**.

---

## Step 5 – Final Evaluation & Decision Freezing

### Running the Evaluation

Final evaluation is executed using the following command:

```bash
python src\evaluation\evaluate_final_model.py
```

This script performs **read-only evaluation** of the frozen model and decision policy.
No training, tuning, or threshold selection occurs at this stage.

---

### Example Output

```
Using artifacts from: artifacts\final_model\20260204_153203
test.parquet not found – reconstructing test split from full dataset
=== Final Test Evaluation ===
Run directory     : artifacts\final_model\20260204_153203
Threshold         : 0.48000000000000004
Precision         : 0.9280
Recall            : 0.9258
F1 Score          : 0.9269
Confusion Matrix:
[[390522   1432]
 [  1479  18450]]
```

---

### Explanation of the Output

#### Artifact Selection

```
Using artifacts from: artifacts\final_model\20260204_153203
```

The evaluation pipeline automatically selects the **most recent finalized training run** under `artifacts/final_model/`.

* Only directories containing `final_model.joblib` are considered
* The latest timestamped run is assumed to be the chosen *final model*

Model comparison or selection does **not** occur in Step 5.

---

#### Metrics

```
Precision : 0.9280
Recall    : 0.9258
F1 Score  : 0.9269
```

* **Precision (0.928)**
  ~93% of alerts correspond to real attacks
  → Low alert noise for analysts

* **Recall (0.9258)**
  ~92.6% of attacks are successfully detected
  → Few attacks are missed

* **F1 Score (0.9269)**
  Strong balance between detection coverage and alert quality

---

#### Confusion Matrix

```
[[390522   1432]
 [  1479  18450]]
```

Interpreted as:

|                   | Predicted Normal | Predicted Attack |
| ----------------- | ---------------- | ---------------- |
| **Actual Normal** | 390,522          | 1,432 (FP)       |
| **Actual Attack** | 1,479 (FN)       | 18,450 (TP)      |

* **False Positives (1,432)**
  Benign flows flagged as attacks (analyst noise)

* **False Negatives (1,479)**
  Attacks missed by the model (security risk)

---


## Step 6 – API Simulation via CLI (Inference & Risk Framing)

This step demonstrates how the trained model and the deterministic risk framing logic can be executed **end-to-end via the command line**, simulating an API-style inference call.

Instead of deploying a web service, we expose the inference logic through a CLI interface that mirrors how an API endpoint would behave:
input → prediction → risk framing → structured JSON output.

---

### Running Inference on a Low-Risk Flow

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/flow_low_risk.json
```

Output:

```json
{
  "risk_score": 0.0,
  "risk_level": "LOW",
  "reasons": []
}
```

**Interpretation:**

* The model assigns an extremely low probability to an attack.
* No risk rules are triggered.
* The flow is classified as clearly benign and produces no alerts or explanations.

---

### Running Inference on a High-Risk Flow

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/flow_high_risk.json
```

Output:

```json
{
  "risk_score": 0.9784,
  "risk_level": "HIGH",
  "reasons": [
    "Unusually long connection duration",
    "Large outbound data transfer",
    "High fan-out to destination ports",
    "High number of connections to the same destination"
  ]
}
```

**Interpretation:**

* The model assigns a very high probability to an attack.
* Multiple deterministic risk rules are triggered based on raw flow features.
* The output includes both a quantitative risk score and human-readable explanations suitable for SOC analysts or downstream systems.

---

## Step 6 (Extended) – API Simulation with SHAP Explanations (CLI)

In addition to standard risk prediction, the CLI supports **optional SHAP-based explainability** for single-flow inference.
This mode is intended for **debugging, investigation, and analyst drill-down**, not for high-throughput batch usage.

SHAP explanations are returned as **Top-5 features by absolute contribution**, ensuring concise and actionable output.

---

### High-Risk Flow with SHAP Explanation

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/flow_high_risk.json --explain shap
```

Output:

```json
{
  "risk_score": 0.9784,
  "risk_level": "HIGH",
  "reasons": [
    "Unusually long connection duration",
    "Large outbound data transfer",
    "High fan-out to destination ports",
    "High number of connections to the same destination"
  ],
  "shap_top_5_values": {
    "sttl": 9.499049674149832,
    "ct_state_ttl": 4.424585564732738,
    "ct_dst_sport_ltm": 1.2731662270240485,
    "ct_srv_src": -0.7075171171852086,
    "destination_port": 0.5992478361381254
  }
}
```

**Interpretation:**

* The model assigns very high confidence to an attack.
* Multiple deterministic risk rules are triggered.
* SHAP highlights the top 5 features that most strongly influenced the prediction, including both positive and negative contributions.
* Explanations combine **domain rules** (why it is risky) with **model-level attribution** (what drove the prediction).

---

### Low-Risk Flow with SHAP Explanation

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/flow_low_risk.json --explain shap
```

Output:

```json
{
  "risk_score": 0.0,
  "risk_level": "LOW",
  "reasons": [],
  "shap_top_5_values": {
    "sttl": -0.6765036614612417,
    "sintpkt": 0.5787860138168665,
    "stime": 0.42347297708016335,
    "ct_state_ttl": -0.315744233391285,
    "smeansz": -0.2278311339103425
  }
}
```

**Interpretation:**

* The model assigns an extremely low probability to an attack.
* No risk rules are triggered.
* SHAP values still show which features contributed most to the benign classification, including features that actively reduce risk.

---

## Step 6 (Extended) – Batch Data Preparation and Inference Validation

This step demonstrates the **end-to-end inference workflow** for both single-flow and batch processing, including explainability and validation of mixed benign/attack inputs.

The goal is to validate that:

* The model produces sensible risk scores across different flows
* Batch inference behaves consistently with single inference
* Rule-based explanations remain conservative and deterministic

---

### 6.1 Creating a Mixed Batch Input (Attack + Benign)

We first generate a batch input file containing a balanced mix of attack and benign flows, sampled directly from the processed UNSW-NB15 dataset.

Command:

```bash
python src/inference/make_batch_flows.py
```

Console output:

```text
Loading latest feature list from final_model artifacts...
Loading processed dataset...
Sampling attack=10, benign=10 ...
Saved: src/inference/examples_api_samples/batch_flows.parquet
Preview label counts (if available):
binary_label
1    10
0    10
```

**Explanation:**

* The script loads the feature schema from the latest trained model run.
* It samples an equal number of attack and benign flows.
* The resulting `batch_flows.parquet` file is guaranteed to be compatible with the inference pipeline.
* Labels are kept only for offline verification; they are ignored during inference.

---

### 6.2 Single-Flow Inference with SHAP (Low-Risk Example)

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/flow_low_risk.json --explain shap
```

Output:

```json
{
  "risk_score": 0.0,
  "risk_level": "LOW",
  "reasons": [],
  "shap_top_5_values": {
    "sttl": -0.6765,
    "sintpkt": 0.5788,
    "stime": 0.4235,
    "ct_state_ttl": -0.3157,
    "smeansz": -0.2278
  }
}
```

**Interpretation:**

* The model assigns an extremely low probability to an attack.
* No deterministic risk rules are triggered.
* SHAP still highlights the most influential features, including both positive and negative contributions, explaining *why* the flow is considered benign.

---

### 6.3 Batch Inference on Mixed Flows

We now run batch inference on the generated parquet file.

Command:

```bash
python src/inference/predict.py --input src/inference/examples_api_samples/batch_flows.parquet --batch
```

Output (excerpt):

```json
[
  {
    "row_index": 0,
    "risk_score": 0.988,
    "risk_level": "HIGH",
    "reasons": []
  },
  {
    "row_index": 1,
    "risk_score": 0.0,
    "risk_level": "LOW",
    "reasons": []
  },
  {
    "row_index": 4,
    "risk_score": 0.5616,
    "risk_level": "MEDIUM",
    "reasons": []
  },
  {
    "row_index": 11,
    "risk_score": 0.9998,
    "risk_level": "HIGH",
    "reasons": []
  }
]
```

**Interpretation:**

* The batch contains a mix of `LOW`, `MEDIUM`, and `HIGH` risk flows, as expected.
* High-risk flows often receive very high confidence scores due to strong learned patterns in the data.
* Most entries have empty `reasons` because:

  * Rule-based explanations are intentionally conservative.
  * Many attacks are detected by the model through feature combinations rather than extreme threshold violations.

---

### 6.4 Design Rationale

* The model is trained on a **binary task only** (`attack` vs. `benign`).
* Risk levels (`LOW`, `MEDIUM`, `HIGH`) are derived from probability thresholds after inference.
* Rule-based explanations are:

  * Deterministic
  * Interpretable
  * Independent of the model
* SHAP explanations are:

  * Optional
  * Available only for single-flow inference
  * Limited to Top-5 features for clarity and performance

This design mirrors real-world production systems where:

* Batch jobs compute large-scale risk snapshots
* APIs handle single events and investigations
* Explainability is applied selectively and on demand

---





