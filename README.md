# LOAN_DEAFULT_PREDICTION

## Project Overview
Loan default prediction is a core problem in consumer lending. The goal is to classify borrowers as Fully Paid or Charged Off (Default) using only features available before a loan is issued — avoiding data leakage from post-issuance payment columns.
Target Variable: loan_status → Binary: Default (Charged Off) vs Fully Paid

## Dataset

Source: LendingClub Accepted Loans (2007–2018 Q4)

Size: ~2.2 million rows, 150 columns (raw)

Note: The dataset is too large for GitHub. See How to Run for download instructions.

## What I Did

1. Loaded 2M+ rows using chunked reading (50k-row batches) to manage memory
2. Prevented data leakage by excluding post-issuance payment columns
3. Data cleaning — dropped high-null columns, capped outliers at 99th percentile, imputed missing values
4. Feature engineering — extracted issue date components, computed credit history length
5. Handled class imbalance using SMOTE (training set only)
6. Trained 5 models: Logistic Regression, Decision Tree, Random Forest, XGBoost (default), XGBoost (Optuna tuned)
7. Tuned XGBoost with Optuna (Bayesian search, 3-fold CV, 5-min timeout)
8. Evaluated all models with ROC-AUC, Precision, Recall, F1-score, and Confusion Matrices

## Key Results

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| Decision Tree | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| Random Forest | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| XGBoost (default) | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| **XGBoost (Optuna tuned)** | **X.XXXX** | **X.XXXX** | **X.XXXX** | **X.XXXX** |

> Best model: **XGBoost (Optuna tuned)** with ROC-AUC of X.XXXX
