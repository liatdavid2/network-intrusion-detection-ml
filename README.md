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
network-intrusion-detection-ml/
├── artifacts/
│   └── models/
│       └── 20260202_143325/
│           ├── ExtraTrees.joblib
│           └── GradientBoosting.joblib
│           └── HistGradientBoosting.joblib
│           └── RandomForest.joblib
│           └── XGBoost.joblib
│ 
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_dataset_familiarization_unsw_nb15.ipynb
│
├── src/
│   ├── build_features.py
│   └── models/
│       ├── train_baseline_models.py
│       └── explain_models_shap.py
│
├── .gitignore
├── README.md
└── requirements.txt

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

