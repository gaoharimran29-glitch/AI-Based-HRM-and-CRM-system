import pandas as pd
import joblib
from feature_engineer import LeadFeatureEngineer

model = joblib.load("catboost_lead_scoring_model2.pkl")

test_scenarios = {
    "1. REALISTIC BUSINESS PERSONAS": pd.DataFrame([
        {
            "industry": "Finance", "budget": 650000, "website_visits": 24, 
            "mail_response_count": 5, "total_meet_count": 3, "mail_open_rate": 0.55
        }, # Expected: Hot Whale
        {
            "industry": "Manufacturing", "budget": 35000, "website_visits": 1, 
            "mail_response_count": 0, "total_meet_count": 0, "mail_open_rate": 0.05
        }  # Expected: Cold Tire-Kicker
    ]),
    
    "2. EXTREME EDGE CASES (Zero-Division & Form Typos)": pd.DataFrame([
        {
            "industry": "Healthcare", "budget": 5000, "website_visits": 0, 
            "mail_response_count": 0, "total_meet_count": 0, "mail_open_rate": 0.0
        }, # Zero values test
        {
            "industry": "Deep-Tech Quantum Startup", "budget": 4500000, "website_visits": 850, 
            "mail_response_count": 120, "total_meet_count": 35, "mail_open_rate": 0.98
        } # Out-of-bounds scale & unseen category name
    ]),
    
    "3. ADVERSARIAL ATTACKS (Hostile Data Errors)": pd.DataFrame([
        {
            "industry": "SaaS", "budget": -75000, "website_visits": 15, 
            "mail_response_count": 2, "total_meet_count": 1, "mail_open_rate": 0.25
        }, # Malicious negative budget
        {
            "industry": "Finance", "budget": 120000, "website_visits": 8, 
            "mail_response_count": -10, "total_meet_count": 0, "mail_open_rate": -0.80
        } # Corrupted data logging (negative activities)
    ])
}

def assign_lead_tier(prob):
    if prob >= 0.70:
        return "🔥 Hot (Route to AE)"
    elif prob >= 0.30:
        return "⚡ Warm (Marketing Nurture)"
    else:
        return "❄️ Cold (Low Priority)"

# =====================================================================
# 3. AUTOMATED TESTING LOOP EXECUTION
# =====================================================================
for test_name, raw_data in test_scenarios.items():
    print(f"\n🚀 RUNNING SCENARIO: {test_name}")
    print("-" * 60)
    
    # Process predictions via pipeline
    try:
        probs = model.predict_proba(raw_data)[:, 1]
        
        # Display human-readable reports
        output_df = raw_data.copy()
        output_df["Prob"] = probs
        output_df["Actionable Tier"] = output_df["Prob"].apply(assign_lead_tier)
        
        # Format printing output
        for idx, row in output_df.iterrows():
            print(f" Lead #{idx} ({row['industry']}) | Budget: ${row['budget']:,}")
            print(f" └─ Activity Metrics -> Meets: {row['total_meet_count']}, Web Visits: {row['website_visits']}")
            print(f" └─ SCORING RESULT   -> Probability: {row['Prob']:.4f} | Tier: {row['Actionable Tier']}\n")
            
    except Exception as e:
        print(f"💥 CRITICAL BREAKDOWN: Pipeline crashed on this block!\nError Details: {str(e)}")
    
    print("=" * 50)