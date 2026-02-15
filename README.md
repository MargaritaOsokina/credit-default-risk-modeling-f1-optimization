Credit Default Risk Modeling Using Machine Learning

1. Introduction

This project aims to develop a machine learning model for predicting credit default based on historical client data. The task is formulated as a binary classification problem where the objective is to determine whether a borrower will fail to meet debt obligations on a current loan.

The primary evaluation metric is the F1-score for the positive class (Credit Default = 1). The project requirement is to achieve F1 > 0.5.

2. Problem Formulation
	•	Target variable: Credit Default
	•	1 — default
	•	0 — no default
	•	Task type: Binary classification
	•	Evaluation metric: F1-score (positive class)

The F1-score is selected due to class imbalance and the need to balance:
	•	Precision (minimizing false approvals of high-risk clients)
	•	Recall (identifying potentially risky borrowers)
  
3. Dataset Description

The training dataset contains 7,500 observations and 17 variables, including:
	•	Demographic characteristics (e.g., Home Ownership)
	•	Income-related indicators (Annual Income)
	•	Credit history variables (Years of Credit History, Credit Score)
	•	Debt and loan characteristics (Monthly Debt, Current Loan Amount, Current Credit Balance)
	•	Delinquency and bankruptcy indicators

The dataset includes:
	•	Numerical and categorical features
	•	Missing values in several variables (e.g., Annual Income, Credit Score, Months since last delinquent)
	•	Moderate class imbalance in the target variable

4. Exploratory Data Analysis

The exploratory data analysis included:
	1.	Examination of class distribution.
	2.	Analysis of missing values.
	3.	Correlation analysis for numerical variables.
	4.	Inspection of distributions of key financial indicators.

Key findings:
	•	The dataset exhibits class imbalance, justifying the use of F1-score.
	•	Financial behavior variables (debt burden, credit balance, credit score, delinquency history) are associated with default risk.
	•	Several predictors contain missing values and require systematic imputation.

5. Methodology

5.1 Train-Validation Split

A stratified sampling strategy was applied using StratifiedShuffleSplit to preserve the distribution of the target variable in both training and validation sets.

5.2 Data Preprocessing

A production-style preprocessing pipeline was implemented using:
	•	SimpleImputer (median strategy for numerical features, most frequent for categorical features)
	•	StandardScaler (for numerical feature scaling)
	•	OneHotEncoder (for categorical encoding)
	•	ColumnTransformer (to combine preprocessing steps)

6. Model Development

6.1 Baseline Model: Logistic Regression

Logistic Regression was implemented as a baseline model with:
	•	Class imbalance handling via class_weight="balanced"
	•	Maximum iteration control for convergence

6.2 Threshold Optimization

Since F1-score depends on the classification threshold, probability threshold tuning was performed on the validation set.

The optimal threshold improved the balance between precision and recall and increased the F1-score compared to the default threshold of 0.5.

6.3 Random Forest Classifier

A Random Forest classifier was implemented to capture non-linear relationships between predictors and default risk.

Hyperparameter tuning was performed using GridSearchCV with cross-validation.

Parameters optimized:
	•	Number of estimators
	•	Maximum tree depth

6.4 Custom Logistic Regression Implementation

A custom implementation of Logistic Regression was developed using:
	•	Sigmoid activation function
	•	Gradient descent optimization
	•	Binary decision threshold

7. Model Evaluation

Models were evaluated using the F1-score for the positive class.

Model	F1-score
Logistic Regression (baseline)	0.xx
Logistic Regression (optimized threshold)	0.xx
Random Forest	0.xx
Tuned Random Forest	0.xx
Custom Logistic Regression	0.xx

The final selected model achieves:

F1-score > 0.5

8. Feature Importance Analysis

Feature importance analysis (based on the Random Forest model) indicates that the most influential predictors include:
	•	Credit Score
	•	Annual Income
	•	Monthly Debt
	•	Current Credit Balance
	•	Loan Amount
	•	Indicators of past delinquency

9. Conclusion

This project presents a complete machine learning workflow for credit default prediction, including:
	•	Business problem formulation
	•	Exploratory data analysis
	•	Preprocessing pipeline construction
	•	Model comparison
	•	Hyperparameter tuning
	•	Threshold optimization
	•	Custom algorithm implementation
