import numpy as np
import pandas as pd

df = pd.read_csv("synthetic_it_services_dataset2.csv")

df.drop(columns=["persona", "conversion_prob"], errors="ignore", inplace=True)

df.columns = [
    "industry",
    "budget",
    "mail_response_count",
    "total_meet_count",
    "mail_open_rate",
    "website_visits",
    "converted",
]

industry_medians = df.groupby("industry")["website_visits"].median()

for industry in df["industry"].unique():
    mask = (df["industry"] == industry) & (df["website_visits"].isnull())
    num_missing = mask.sum()

    if num_missing > 0:
        median_val = industry_medians[industry]
        noise = np.random.randint(-2, 3, size=num_missing)

        df.loc[mask, "website_visits"] = np.maximum(0, median_val + noise)


df["mail_open_rate"] = df["mail_open_rate"].fillna(df["mail_open_rate"].median())

df = df.drop_duplicates()

print("--- Final Dataset Sanity Check ---")
print(f"Dataset Shape: {df.shape}")
print(f"Remaining Missing Values:\n{df.isnull().sum()}")

df.to_csv("synthetic_it_services_dataset.csv", index=False)