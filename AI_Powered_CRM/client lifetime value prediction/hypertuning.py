import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

# -------- LOAD DATA --------
file_path = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\cleaned_dataset.csv"
df = pd.read_csv(file_path)

# -------- TARGET --------
y = np.log1p(df['Monetary'])

# -------- FEATURES --------
X = df.drop(columns=['Monetary', 'Customer ID'])

feature_names = X.columns  # save for feature importance

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- PIPELINE --------
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# -------- PARAM GRID --------
param_dist = {
    'model__n_estimators': [200, 300, 400, 500],
    'model__max_depth': [8, 10, 12, 15, None],
    'model__min_samples_split': [2, 4, 6, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2', None]
}

# -------- RANDOM SEARCH --------
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

# -------- PREDICTION --------
y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# -------- METRICS --------
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

print("\n--- FINAL RESULTS ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# -------- SAVE MODEL --------
model_dir = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\saved_model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "clv_model_tuned.pkl")
joblib.dump(best_model, model_path)

print(f"\n✓ Model saved at: {model_path}")

# -------- PLOTS --------
plot_dir = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\plots"
os.makedirs(plot_dir, exist_ok=True)

# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test_actual, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig(os.path.join(plot_dir, "actual_vs_predicted.png"))
plt.close()

# 2. Residual Plot
residuals = y_test_actual - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig(os.path.join(plot_dir, "residual_plot.png"))
plt.close()

# 3. Residual Distribution
plt.figure()
plt.hist(residuals)
plt.title("Residual Distribution")
plt.savefig(os.path.join(plot_dir, "residual_distribution.png"))
plt.close()

# 4. Feature Importance
model = best_model.named_steps['model']

importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
plt.close()

print("\n✓ All plots saved in plots folder!")

"""
--- FINAL RESULTS ---
MAE: 477.74
RMSE: 1414.33
R2 Score: 0.9021
MAPE: 39.44%"""