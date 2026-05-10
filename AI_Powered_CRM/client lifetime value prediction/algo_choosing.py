import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------- LOAD DATA --------
file_path = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\cleaned_dataset.csv"
df = pd.read_csv(file_path)

# -------- TARGET --------
y = np.log1p(df['Monetary'])

# -------- FEATURES --------
X = df.drop(columns=['Monetary', 'Customer ID'])

# -------- SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- MODELS --------
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
}

best_model = None
best_r2 = -np.inf

# -------- TRAIN + EVALUATE --------
for name, model in models.items():

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # real scale metrics
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)

    # log scale metrics
    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))

    # business metric
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

    print(f"\nModel: {name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Log MAE: {mae_log:.4f}")
    print(f"Log RMSE: {rmse_log:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("-" * 40)

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

# -------- SAVE MODEL --------
output_dir = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data"
model_dir = os.path.join(output_dir, "saved_model")
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.join(output_dir, "saved_model")

print("\n✓ Best Model Saved!")
print(f"Best R2: {best_r2:.4f}")

""" 
Model: RandomForest
MAE: 481.39
RMSE: 1430.79
R2 Score: 0.8999
Log MAE: 0.2992
Log RMSE: 0.4426
MAPE: 39.70%
----------------------------------------

Model: XGBoost
MAE: 498.00
RMSE: 1591.16
R2 Score: 0.8761
Log MAE: 0.2935
Log RMSE: 0.4166
MAPE: 34.12%
----------------------------------------

✓ Best Model Saved!
Best R2: 0.8999 """