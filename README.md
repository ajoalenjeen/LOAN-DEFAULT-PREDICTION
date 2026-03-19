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

* XGBoost (Optuna tuned): ROC-AUC ≈ 0.96 | Accuracy ≈ 86% (Recall ≈ 0.92) ← Best model
* XGBoost (default): ROC-AUC ≈ 0.96 | Accuracy ≈ 85% (Recall ≈ 0.94)
* Decision Tree: ROC-AUC ≈ 0.93 | Accuracy ≈ 86% (Recall ≈ 0.83)
* Random Forest: ROC-AUC ≈ 0.93 | Accuracy ≈ 87% (Recall ≈ 0.76)
* Logistic Regression: ROC-AUC ≈ 0.93 | Accuracy ≈ 88% (Recall ≈ 0.61)


## ▶ How to Run

### 1. Clone the repo
git clone https://github.com/ajoalenjeen/LOAN_DEFAULT_PREDICTION.git
cd LOAN_DEFAULT_PREDICTION

### 2. Create virtual environment and install dependencies
python -m venv venv

# Activate the environment
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install required packages
pip install -r requirements.txt

### 3. Run the notebook
jupyter lab notebooks/loan_default_prediction.ipynb
