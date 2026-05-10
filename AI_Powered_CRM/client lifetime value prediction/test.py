import pandas as pd
import numpy as np
import joblib

# -------- LOAD MODEL --------
model_path = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\saved_model\clv_model_tuned.pkl"
model = joblib.load(model_path)

print("✓ Model Loaded Successfully!")

# -------- SAMPLE INPUT --------
sample_data = {
    "Recency": [30],
    "Frequency": [5],
    "Unique_Products": [40],
    "Customer_Lifetime_Days": [200],
    "Expansion_Velocity": [0.2],
    "Purchase_Consistency": [10],
    "Items_Per_Order": [150],
    "Purchase_Frequency_Rate": [0.025],
    "Log_Frequency": [np.log1p(5)],
    "Log_Unique_Products": [np.log1p(40)]
}

df_sample = pd.DataFrame(sample_data)

# -------- PREDICT CLV --------
pred_log = model.predict(df_sample)
pred_clv = np.expm1(pred_log)[0]

if pred_clv < 500:
    upsell = "Low"
elif pred_clv < 2000:
    upsell = "Medium"
else:
    upsell = "High"

expansion_velocity = df_sample['Expansion_Velocity'].values[0]

cross_sell_days = 1 / (expansion_velocity + 1e-5)

# -------- FINAL OUTPUT --------
print("\n--- CRM INSIGHTS ---")
print(f"Predicted CLV: {pred_clv:.2f}")
print(f"Upsell Opportunity: {upsell}")
print(f"Estimated Cross-sell Timing: {cross_sell_days:.1f} days")