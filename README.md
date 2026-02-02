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
├── src/
│   ├── data/        # Data loading and validation
│   ├── features/    # Feature engineering
│   ├── models/      # Training and evaluation
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

````md
Run feature engineering:
```bash
python src/build_features.py
````
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


