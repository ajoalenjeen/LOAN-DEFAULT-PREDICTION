# LOAN-DEFAULT-PREDICTION
## Overview

This project builds an end-to-end loan default prediction model to classify whether a loan will be Fully Paid or Charged Off using borrower financial and credit attributes. The goal is to identify key risk drivers and compare multiple machine learning models using robust evaluation metrics.

## Dataset:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Methodology and Steps Taken
1. Data Loading and Initial Inspection

* The LendingClub accepted loans dataset was loaded into a Pandas DataFrame.

* Initial inspection revealed:

      * ~1.3 million completed loan records (Fully Paid / Charged Off)

      * Multiple borrower, credit, and loan-level attributes

      * Significant missingness in certain hardship and joint application fields

* A binary target variable default_flag was created: 1 → Charged Off ,  0 → Fully Paid

* The dataset showed class imbalance: ~80% Fully Paid  ,  ~20% Charged Off

## The project covers the complete analytics workflow:

Data preparation and cleaning

Exploratory data analysis (EDA)

Feature engineering and preprocessing

Predictive modeling and hyperparameter tuning

Model evaluation and comparison

## Exploratory Data Analysis (EDA)

EDA was conducted to understand default behavior across borrower and loan characteristics.

Key analyses include:

* Loan status distribution

Default rates by:

* Loan grade and sub-grade

* Loan term (36 vs 60 months)

* Home ownership

* Verification status

* Employment length

* Loan purpose

* FICO score bands

* Debt-to-Income (DTI) bands

* Income bands

Key findings:

* Default rates increase as FICO score decreases

* Higher DTI and lower income bands show elevated default risk

## Feature Engineering & Preprocessing

* Removed redundant and low-correlation features (|corr| < 0.03)

* Dropped high-cardinality identifiers (id, emp_title, title, zip_code)

* Converted loan term to numeric values

* Extracted year from earliest_cr_line

* Log-transformed income (log_annual_inc)

* Created loan amount to installment ratio

* Label-encoded categorical variables

* Applied mean imputation for missing values

Final modeling dataset:

* 27 features + target

* Fully numeric and model-ready

* Riskier loan purposes (e.g., small business, medical) have higher default rates

* Employment length shows minimal variation and was removed


## Modeling Approach
* Class Balancing

* To address class imbalance, undersampling was applied:

* 10,000 defaulted loans

* 10,000 non-defaulted loans

* Final balanced dataset: 20,000 observations

* Train/Test Split (80% training set, 20% testing set)

* Fixed random seed for reproducibility

## Models Trained

All models were tuned using GridSearchCV with ROC-AUC as the primary metric.

Models evaluated:

* Logistic Regression

* Random Forest

* Gradient Boosting

* XGBoost

Each model was evaluated using:

* Accuracy

* Precision, Recall, F1-score

* Confusion matrix

* Cross-validated ROC-AUC

## Model Performance

| Model                 | CV ROC-AUC |
| --------------------- | ---------- |
| **Gradient Boosting** | **0.9479** |
| XGBoost               | 0.9467     |
| Logistic Regression   | 0.9283     |
| Random Forest         | 0.9225     |

Gradient Boosting achieved the best overall performance with strong generalization on unseen data.
