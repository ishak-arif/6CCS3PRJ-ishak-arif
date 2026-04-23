# Evaluating the Consistency of SHAP-Based Explanations Across Unsupervised Anomaly Detectors for Credit Card Fraud Detection

BSc Final Year Project — King's College London

## Requirements

Python 3.11 and the following libraries:

```
pip install tensorflow scikit-learn shap numpy pandas scipy matplotlib
```

## Dataset

Download the dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in the `data/` folder.

## Running the Pipeline

```
python main.py
```

This trains both models, computes SHAP explanations, runs all validation checks, and exports all outputs.

## Outputs

- `outputs/results/` — 13 CSV files
- `outputs/plots/` — 13 figures

## Configuration

Parameters are defined as named constants at the top of `pipeline.py`.