# Credit Default Prediction

## Project Overview

This project focuses on predicting credit default (non-payment of debt obligations) using machine learning models. The goal is to build and compare various classification models to identify clients at risk of defaulting on their current loans.

## Problem Statement

Using historical client data, we need to build predictive models (Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting) to forecast credit default. The models are trained on the provided training dataset and evaluated on a test dataset.

## Dataset Description

### Target Variable

- **Credit Default**: Binary indicator (1 - default occurs, 0 - no default)

### Features

- **Home Ownership**: Type of housing ownership (Own Home, Rent, Home Mortgage)
- **Annual Income**: Yearly income of the borrower
- **Years in current job**: Years employed at current workplace
- **Tax Liens**: Presence of tax obligations or debts
- **Number of Open Accounts**: Total number of open accounts (credit cards, credit lines, etc.)
- **Years of Credit History**: Total duration of credit history in years
- **Maximum Open Credit**: Maximum credit line provided to borrower
- **Number of Credit Problems**: Count of credit issues (late payments, collections)
- **Months since last delinquent**: Months since last payment default
- **Bankruptcies**: Number of bankruptcies in credit history
- **Purpose**: Loan purpose (debt consolidation, etc.)
- **Term**: Loan term (Short Term/Long Term)
- **Current Loan Amount**: Current outstanding loan amount
- **Current Credit Balance**: Current credit balance across all accounts
- **Monthly Debt**: Total monthly debt payments
- **Credit Score**: Credit score indicating creditworthiness

## Solution Structure

### 1. Exploratory Data Analysis (EDA)

- Target variable distribution analysis
- Missing values identification and visualization
- Numerical features distribution analysis
- Correlation analysis with heatmaps
- Categorical features distribution analysis

### 2. Data Preprocessing

- Stratified train/validation split (80/20)
- Numerical features: Median imputation + StandardScaler
- Categorical features: Most frequent imputation + OneHotEncoder
- ColumnTransformer for unified preprocessing

### 3. Modeling

**Models Implemented:**

- Logistic Regression (baseline)
- Decision Tree
- Random Forest (with hyperparameter tuning)
- XGBoost

### 4. Model Evaluation

**Metrics Used:**

- **F1-score**: Primary metric (> 0.5 required)
- **ROC-AUC**: Area Under the ROC Curve
- **Gini**: Calculated as 2*ROC-AUC - 1

### 5. Hyperparameter Tuning

GridSearchCV for Random Forest:

- `n_estimators`: [200, 400]
- `max_depth`: [4, 6, 8]
- Scoring: F1-score with 5-fold cross-validation

## Results

| Model | F1 Score | ROC-AUC | Gini |
|-------|----------|---------|------|
| Random Forest | **0.531** | **0.762** | **0.523** |
| Decision Tree | 0.525 | 0.740 | 0.479 |
| Logistic Regression | 0.517 | 0.754 | 0.508 |
| XGBoost | 0.444 | 0.744 | 0.487 |

**Best Model:** Random Forest achieved the highest F1-score of 0.531, meeting the project requirement of F1 > 0.5.

## Key Insights

1. **Class Imbalance**: The dataset shows class imbalance with ~28% default cases, requiring balanced class weights in modeling
2. **Missing Values**: Several features contain missing values, handled through appropriate imputation strategies
3. **Feature Importance**: Credit Score, Current Loan Amount, and Annual Income show strong correlation with default
4. **Model Performance**: Ensemble methods (Random Forest) outperformed simpler models, though all except XGBoost met the F1 > 0.5 threshold

## Conclusion

The Random Forest model with tuned hyperparameters provides the best performance for predicting credit default, achieving an F1-score of 0.531. The complete ML pipeline includes thorough EDA, preprocessing, modeling, and evaluation stages, meeting all project requirements.
