import numpy as np
import pandas as pd

np.random.seed(42)
num_rows = 10000

# ==========================================
# STEP 1: INDUSTRIES & PERSONAS
# ==========================================
industries = ["Finance", "Healthcare", "SaaS", "E-commerce", "Manufacturing"]
industry_col = np.random.choice(
    industries, size=num_rows, p=[0.25, 0.20, 0.30, 0.15, 0.10]
)

personas = ["Enterprise Whale", "Tire Kicker", "Ghost Lead", "Standard Mid-Market"]
persona_probs = [0.20, 0.15, 0.15, 0.50]
persona_col = np.random.choice(personas, size=num_rows, p=persona_probs)

budget_col = np.zeros(num_rows, dtype=float)
open_rate_col = np.zeros(num_rows)
response_col = np.zeros(num_rows, dtype=int)
meet_col = np.zeros(num_rows, dtype=int)
web_visits_col = np.zeros(num_rows, dtype=int)
base_conversion_intent = np.zeros(num_rows)

# ==========================================
# STEP 2: CAUSAL FEATURE GENERATION
# ==========================================
for i in range(num_rows):
    p = persona_col[i]
    ind = industry_col[i]

    if ind in ["Finance", "Healthcare"]:
        ind_budget_scale = 1.3
    elif ind == "Manufacturing":
        ind_budget_scale = 0.9
    else:
        ind_budget_scale = 1.0

    if p == "Enterprise Whale":
        budget_col[i] = np.random.lognormal(mean=12.2, sigma=0.4) * ind_budget_scale
        open_rate_col[i] = np.random.beta(a=3, b=8)
        response_col[i] = np.random.poisson(lam=max(open_rate_col[i] * 6, 1))
        meet_col[i] = np.random.poisson(lam=1.5 * response_col[i]) if response_col[i] > 0 else 0
        web_visits_col[i] = np.random.negative_binomial(n=3, p=0.3)
        base_conversion_intent[i] = -0.5 + (1.5 * meet_col[i]) + (0.3 * response_col[i])

    elif p == "Tire Kicker":
        budget_col[i] = np.random.lognormal(mean=9.5, sigma=0.3) * ind_budget_scale
        open_rate_col[i] = np.random.beta(a=8, b=4)
        response_col[i] = np.random.poisson(lam=1.5)
        meet_col[i] = np.random.choice([0, 1], p=[0.85, 0.15])
        web_visits_col[i] = np.random.negative_binomial(n=8, p=0.15)
        base_conversion_intent[i] = -2.5 + (0.3 * meet_col[i]) + (0.05 * web_visits_col[i])

    elif p == "Ghost Lead":
        budget_col[i] = np.random.lognormal(mean=10.8, sigma=0.5) * ind_budget_scale
        open_rate_col[i] = np.random.beta(a=4, b=8)
        response_col[i] = np.random.poisson(lam=3.0)
        meet_col[i] = np.random.poisson(lam=2.0)
        web_visits_col[i] = np.random.negative_binomial(n=4, p=0.2)
        base_conversion_intent[i] = -3.0 + (0.8 * meet_col[i]) + (0.2 * response_col[i])

    else:  # Standard Mid-Market
        budget_col[i] = np.random.lognormal(mean=10.8, sigma=0.6) * ind_budget_scale
        open_rate_col[i] = np.random.beta(a=3, b=9)
        response_col[i] = np.random.poisson(lam=2.0 * (open_rate_col[i] * 10))
        meet_col[i] = np.random.poisson(lam=0.6 * response_col[i])
        web_visits_col[i] = np.random.negative_binomial(n=5, p=0.25) + (2 * meet_col[i])
        base_conversion_intent[i] = (
            -1.5 + (0.8 * meet_col[i]) + (0.3 * response_col[i]) + (0.5 * open_rate_col[i] * 10)
        )

budget_col = np.clip(budget_col, 5000, 1200000)

df = pd.DataFrame({
    "persona": persona_col,  # Added to retain grounding verification during EDA
    "industry": industry_col,
    "budget": budget_col,
    "mail_response_count": response_col,
    "total_meet_count": meet_col,
    "mail_open_rate": open_rate_col,
    "website_visits": web_visits_col,
})

# ==========================================
# STEP 3: TARGET GENERATION
# ==========================================
log_budget = np.log1p(df["budget"])
norm_budget = (log_budget - log_budget.mean()) / log_budget.std()

# Corrected array logic alignment
has_meetings = np.where(df["total_meet_count"] > 0, 1, 0)
final_latent_y = base_conversion_intent + (0.5 * norm_budget * has_meetings) - 2.5
# Add noise for soft target boundary
noise = np.random.normal(0, 0.8, size=num_rows)
final_latent_y += noise

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Split into explicit probability metrics vs binary outcome classifications
df["conversion_prob"] = sigmoid(final_latent_y)
df["converted"] = np.where(df["conversion_prob"] >= 0.5, 1, 0)

# ==========================================
# STEP 4: DIRTY DATA (Simulated Anomalies)
# ==========================================
# Inject true structural missingness to test imputation models
df.loc[df.sample(frac=0.05, random_state=42).index, "website_visits"] = np.nan
df.loc[df.sample(frac=0.03, random_state=24).index, "mail_open_rate"] = np.nan 

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nTarget Distribution:\n", df["converted"].value_counts(normalize=True))

# Exporting cleanly
df.to_csv("synthetic_it_services_dataset2.csv", index=False)