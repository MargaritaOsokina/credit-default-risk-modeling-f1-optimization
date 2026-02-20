# Credit default prediction project

## Project description

The objective of this project is to build machine learning models to predict credit default based on client financial and demographic information.

The task is formulated as a binary classification problem where:
- **1** — client defaults on the loan
- **0** — client does not default

Models are trained on the training dataset and evaluated using validation data. Final predictions are generated for the test dataset.

## Dataset description

The dataset contains information about borrowers, including both numerical and categorical features.

**Features:**
- Home ownership
- Annual income
- Years in current job
- Tax liens
- Number of open accounts
- Years of credit history
- Maximum open credit
- Number of credit problems
- Months since last delinquent
- Bankruptcies
- Purpose of loan
- Term
- Current loan amount
- Current credit balance
- Monthly debt
- Credit score

**Target variable:**
- **Credit Default** — indicates whether the borrower defaulted on the loan.

## Evaluation metrics

The following metrics were used to evaluate model performance:
- **F1-score** (primary metric for the imbalanced default class)
- **ROC-AUC**
- **Gini coefficient**

The Gini coefficient was calculated as:

\[
\text{Gini} = 2 \times \text{ROC-AUC} - 1
\]

The main requirement of the project is:

> **F1-score > 0.5 for the default class (class 1).**

## Exploratory data analysis

During the exploratory data analysis (EDA), the following steps were performed:
- Analysis of dataset shape and structure
- Target distribution analysis (class imbalance detection)
- Missing values analysis
- Distribution analysis of numerical features
- Correlation matrix analysis
- Categorical feature distribution analysis
- Relationship between features and target variable

The dataset shows **class imbalance**, which required the use of class weighting in the models. Several numerical features contain outliers, which were handled implicitly by tree-based models.

## Data preprocessing

The following preprocessing steps were applied:
- **Missing values imputation**:
  - Median strategy for numerical features
  - Most frequent strategy for categorical features
- **Standard scaling** for numerical variables
- **One-hot encoding** for categorical variables
- **Stratified train-validation split** to preserve class distribution

All transformations were implemented using a `sklearn` **Pipeline** and `ColumnTransformer` to ensure reproducibility and prevent data leakage.

## Models used

The following models were implemented and compared:
1. **Logistic regression** (baseline model)
2. **Decision tree**
3. **Random forest**
4. **XGBoost** (gradient boosting)

Hyperparameter tuning was performed for the Random Forest model using `GridSearchCV`.

## Model comparison

All models were evaluated using the metrics mentioned above. Below is a comparison of their performance:

| Model               | F1-score (class 1) | ROC-AUC | Gini |
|---------------------|---------------------|---------|------|
| Logistic Regression | 0.68                 | 0.84     | 0.68  |
| Decision Tree       | 0.71                 | 0.86     | 0.72  |
| Random Forest       | 0.77                 | 0.91     | 0.82  |
| XGBoost             | 0.79                 | 0.93     | 0.86  |

Tree-based ensemble models demonstrated better performance compared to the linear baseline model.

## Final model

The final model was selected based on the highest **F1-score** on the validation dataset. The chosen model was used to generate predictions for the test dataset, and a submission file was created containing predictions for all test observations.

## Key conclusions

- Class imbalance significantly affects model performance.
- Using `class_weight` improved recall for the default class.
- Ensemble methods outperformed logistic regression.
- Gradient boosting showed strong and stable performance.

The project demonstrates a complete machine learning pipeline, including EDA, preprocessing, model development, evaluation, and final prediction generation.

## Results

> 🟢 The final F1-score for the default class (class 1) exceeded the required threshold of **0.5**, satisfying the main project criterion.
