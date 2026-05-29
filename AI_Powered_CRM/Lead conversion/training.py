import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from catboost import CatBoostClassifier
import joblib
from sklearn.pipeline import Pipeline
from feature_engineer import LeadFeatureEngineer

df = pd.read_csv("synthetic_it_services_dataset.csv")

X = df[['industry', 'budget', 'website_visits', 'mail_response_count', 'total_meet_count', 'mail_open_rate']]
y = df['converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tree_preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['industry']),
    remainder='passthrough'
)

tuning_pipeline = Pipeline([
    ('feature_engineer', LeadFeatureEngineer()),
    ('preprocessor', tree_preprocessor),
    ('classifier', CatBoostClassifier(
        verbose=0, 
        random_state=42,
        auto_class_weights='Balanced'
    ))
])

param_grid = {
    'classifier__iterations': [300],
    'classifier__learning_rate': [0.03, 0.1],
    'classifier__depth': [4, 6],
    'classifier__l2_leaf_reg': [3, 5]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting Hyperparameter Tuning...")
grid_search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
y_test_pred = best_pipeline.predict(X_test)
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"\nBest Params: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"Test F1: {f1_score(y_test, y_test_pred):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred))

joblib.dump(best_pipeline, "catboost_lead_scoring_model.pkl")
print("\nSaved fully integrated pipeline successfully!")