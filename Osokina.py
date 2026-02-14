import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Data Loading
def load_data():
    train = pd.read_csv("course_project_train.csv")
    test = pd.read_csv("course_project_test.csv")
    return train, test

train, test = load_data()

train.head()
train.info()

#EDA
#Target Distribution
train['Credit Default'].value_counts(normalize=True)
sns.countplot(x='Credit Default', data=train)
plt.title("Target Distribution")
plt.show()

#Missing Value
train.isna().sum().sort_values(ascending=False)

#Correlation
plt.figure(figsize=(12,8))
sns.heatmap(train.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#Train / Validation Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, valid_idx in split.split(train, train["Credit Default"]):
    strat_train = train.loc[train_idx]
    strat_valid = train.loc[valid_idx]

X_train = strat_train.drop("Credit Default", axis=1)
y_train = strat_train["Credit Default"]

X_valid = strat_valid.drop("Credit Default", axis=1)
y_valid = strat_valid["Credit Default"]

#Preprocessing Pipeline
num_features = X_train.select_dtypes(include=['int64','float64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

#Baseline Model – Logistic Regression
log_model = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ))
])

log_model.fit(X_train, y_train)

preds = log_model.predict(X_valid)

print("Baseline F1:", f1_score(y_valid, preds))
print(classification_report(y_valid, preds))

#Threshold Optimization 
probs = log_model.predict_proba(X_valid)[:,1]

best_f1 = 0
best_t = 0

for t in np.arange(0.1, 0.9, 0.02):
    preds = (probs >= t).astype(int)
    score = f1_score(y_valid, preds)
    if score > best_f1:
        best_f1 = score
        best_t = t

print("Best threshold:", best_t)
print("Best F1:", best_f1)

#Random Forest (Stronger Model)
rf_model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_valid)

print("RF F1:", f1_score(y_valid, rf_preds))

#Hyperparameter Tuning
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [4, 6, 8]
}

grid = GridSearchCV(
    rf_model,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best F1:", f1_score(y_valid, best_model.predict(X_valid)))

#Feature Importance
importances = best_model.named_steps["model"].feature_importances_

feature_names = (
    num_features.tolist() +
    list(best_model.named_steps["prep"]
         .named_transformers_["cat"]
         .named_steps["encoder"]
         .get_feature_names_out(cat_features))
)

feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

feat_importance.head(10)

#Custom Logistic Regression (Advanced Level)
class CustomLogisticRegression:
    
    def __init__(self, lr=0.01, n_iter=3000):
        self.lr = lr
        self.n_iter = n_iter
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(X.shape[1])
        
        for _ in range(self.n_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        probs = self.sigmoid(np.dot(X, self.theta))
        return (probs >= 0.5).astype(int)
    
    X_train_prepared = preprocessor.fit_transform(X_train)
X_valid_prepared = preprocessor.transform(X_valid)

custom_model = CustomLogisticRegression()
custom_model.fit(X_train_prepared, y_train.values)

custom_preds = custom_model.predict(X_valid_prepared)

print("Custom Logistic F1:", f1_score(y_valid, custom_preds))

#Final Prediction for Test
final_probs = best_model.predict_proba(test)[:,1]
final_preds = (final_probs >= best_t).astype(int)

submission = pd.DataFrame({
    "Id": test.index,
    "Credit Default": final_preds
})

submission.to_csv("submission.csv", index=False)