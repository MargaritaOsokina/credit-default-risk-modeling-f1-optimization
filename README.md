# Credit Default Prediction - Machine Learning Project
This project focuses on building a robust machine learning model to predict the likelihood of a borrower defaulting on a loan (credit default). It's a classic binary classification problem in the financial sector. The workflow covers everything from initial data exploration and feature engineering to training, evaluating, and comparing multiple models, including custom implementations of core algorithms.

## 📖 Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology & Workflow](#methodology--workflow)
- [Key Features Engineered](#key-features-engineered)
- [Models Used](#models-used)
- [Evaluation & Results](#evaluation--results)
- [Getting Started](#getting-started)
- [Conclusion](#conclusion)

##  Project Goal
The primary objective is to develop a predictive model that accurately identifies loan applicants who are at a high risk of default. This is a crucial task for financial institutions to manage risk and make informed lending decisions. The main evaluation metric is the **F1-score**, which provides a balanced measure of precision and recall, especially important given the inherent class imbalance in such datasets.

##  Dataset
The dataset contains anonymized information about loan applicants. The training set (`course_project_train.csv`) includes the target variable `Credit Default` (0 for no default, 1 for default), while the test set (`course_project_test.csv`) is provided without labels for final predictions.

**Features include:**
*   **Demographic & Financial:** `Home Ownership`, `Annual Income`
*   **Credit History:** `Credit Score`, `Years of Credit History`, `Number of Credit Problems`, `Bankruptcies`, `Months since last delinquent`
*   **Loan Details:** `Purpose`, `Term`, `Current Loan Amount`
*   **Account Information:** `Number of Open Accounts`, `Maximum Open Credit`, `Current Credit Balance`, `Monthly Debt`, `Tax Liens`

##  Methodology & Workflow
The project follows a structured machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):**
    *   Analyzed target variable distribution, revealing a **class imbalance** (approx. 28% default rate).
    *   Visualized feature distributions and identified significant outliers (e.g., in `Current Loan Amount`).
    *   Examined feature correlations to check for multicollinearity.
    *   Analyzed categorical features and identified missing values.

2.  **Data Preprocessing:**
    *   **`StratifiedShuffleSplit`** was used to create a validation set while preserving the class distribution of the target variable.
    *   A `ColumnTransformer` with separate `Pipeline`s was built for numerical and categorical data to prevent data leakage.
    *   **Numerical Pipeline:** `SimpleImputer` (median) → `PowerTransformer` (Yeo-Johnson) → `StandardScaler`.
    *   **Categorical Pipeline:** `SimpleImputer` (most frequent) → `OneHotEncoder`.

3.  **Feature Engineering:**
    *   New, highly informative features were created to improve model performance (see section below).

4.  **Modeling & Evaluation:**
    *   Trained and evaluated a diverse set of models, including custom-built algorithms for educational purposes.
    *   Used **F1-score** as the primary metric due to class imbalance, alongside **ROC-AUC** and **Gini**.
    *   Performed hyperparameter tuning using **`GridSearchCV`** for the scikit-learn Random Forest.

5.  **Final Prediction:**
    *   The best-performing model was selected to generate predictions on the unseen test set and saved to a submission file.

##  Key Features Engineered
New features were created to capture complex relationships in the data:
*   **`DTI` (Debt-to-Income):** `Monthly Debt * 12 / Annual Income`. A standard measure of financial health.
*   **`Negative_Records`:** Sum of `Bankruptcies`, `Number of Credit Problems`, and `Tax Liens`. A simple aggregate of negative credit events.
*   **`Credit_Utilization`:** `Current Credit Balance / Maximum Open Credit`. Indicates how much of their available credit the borrower is using.
*   **`Loan_to_Income`:** `Current Loan Amount / Annual Income`. Another measure of debt burden relative to income.
*   **`Score_Category`:** Categorizes the `Credit Score` into 'Poor', 'Fair', 'Good', and 'Excellent' buckets.

##  Models Used
The project implemented and compared the following classifiers:
*   **Baseline:** `LogisticRegression`
*   **Tree-Based:** `DecisionTreeClassifier`
*   **Ensemble:** `RandomForestClassifier` (from scikit-learn)
*   **Gradient Boosting:** `XGBClassifier`
*   **Custom Implementations:**
    *   `MyRandomForest`: A Random Forest built from scratch using custom decision trees.
    *   `MyGradientBoostingClassifier`: A Gradient Boosting model built from scratch using scikit-learn's regression trees.

##  Evaluation & Results
The models were evaluated on a held-out validation set. Here is a summary of their performance:

| Model                 |   F1-Score |   ROC-AUC |   Gini |
|:----------------------|-----------:|----------:|-------:|
| **Random Forest (sklearn)** |   0.5365   |   0.7573  | 0.5147 |
| Logistic Regression   |   0.5238   |   0.7470  | 0.4940 |
| Decision Tree         |   0.5132   |   0.7423  | 0.4847 |
| XGBoost               |   0.4363   |   0.7609  | 0.5219 |
| Custom GBM            |   0.4101   |   0.7568  | 0.5136 |

##  Conclusion
This project successfully demonstrates a complete end-to-end machine learning workflow for credit default prediction. Through careful feature engineering and model comparison, a tuned **Random Forest classifier** was identified as the most effective model based on the F1-score. The notebook also showcases a deep understanding of algorithms by implementing custom versions of Random Forest and Gradient Boosting.