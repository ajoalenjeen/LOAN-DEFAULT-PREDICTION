# LOAN-DEFAULT-PREDICTION
## Overview

This project focuses on building a machine learning model to predict loan approval status based on various applicant and loan-related features. The goal is to assist financial institutions in making informed decisions regarding loan applications, thereby reducing risk and improving efficiency in the loan approval process.

* Dataset link [here](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Problem Statement

In the financial industry, accurately assessing the creditworthiness of loan applicants is crucial to minimize financial losses due to defaults. Manual assessment can be time-consuming and prone to human error. There is a need for an automated system that can reliably predict whether a loan application will be approved or rejected based on historical data.

## Methodology and Steps Taken
### 1. Data Loading and Initial Inspection

* The LendingClub accepted loans dataset was loaded into a Pandas DataFrame.

* Initial inspection revealed:

  * ~1.3 million completed loan records (Fully Paid / Charged Off)

  * Multiple borrower, credit, and loan-level attributes

  * Significant missingness in certain hardship and joint application fields

* A binary target variable default_flag was created : 1 → Charged Off ,  0 → Fully Paid

* The dataset showed class imbalance : ~80% Fully Paid  ,  ~20% Charged Off

* Features with correlation < 0.03 were removed to Reduce noise, Improve model efficiency, Avoid unnecessary dimensionality

### 2. Data Preprocessing and Cleaning

* Columns such as id, emp_title, title, zip_code were dropped due to extremely high uniqueness and low predictive value, which could lead to overfitting.

* Handling Missing Values

  * Features with >1,000,000 missing values were removed.

  * Remaining missing numerical values were imputed using column means (for modeling subset).

  * Certain categorical fields were encoded after ensuring no null values remained.

### 3. Exploratory Data Analysis (EDA)

Several visualizations and grouped analyses were conducted to understand default patterns:

* Loan Grade : Default risk increases steadily from Grade A to Grade G, validating LendingClub’s risk grading system.

* Loan Term : 60-month loans exhibit higher default likelihood than 36-month loans.

* Loan Purpose : Highest default rates observed in Small Business, Renewable Energy, Moving, Medical, Debt Consolidation

* An inverse relationship was observed : Higher FICO → Lower default rate

* Default probability increases significantly as DTI moves from : Very Low → Very High

* Income : Higher income bands show progressively lower default rates.

* Credit History Indicators : Public records and bankruptcies are associated with higher default probability.

### 4. Class Imbalance Handling

* Since the dataset was imbalanced (~80/20), a balanced training dataset was created:

  * 10,000 Fully Paid loans

  * 10,000 Charged Off loans

Combined and shuffled to form a 20,000-row balanced dataset

### 5. Model Training:

* The dataset was split into training (80%) and testing (20%) sets.

* Four models were trained using GridSearchCV (3-fold cross-validation) with ROC-AUC scoring:

  * Logistic Regression

  * Random Forest

  * Gradient Boosting

  * XGBoost
 
Hyperparameters were tuned to optimize performance.

### 6. Model Evaluation

| Model                 | CV ROC-AUC |
| --------------------- | ---------- |
| **Gradient Boosting** | **0.9479** |
| XGBoost               | 0.9467     |
| Logistic Regression   | 0.9283     |
| Random Forest         | 0.9225     |

Gradient Boosting achieved the best overall performance with strong generalization on unseen data.

