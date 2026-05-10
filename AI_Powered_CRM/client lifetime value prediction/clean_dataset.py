import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Note: dataset did not contain industry/engagement columns.
# Purchase_Frequency_Rate and Expansion_Velocity used as proxies.

file_path = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data\online_retail_dataset.xlsx"
df = pd.read_excel(file_path)

def get_final_b2b_matrix(df):

    df = df.dropna(subset=['Customer ID'])
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalRevenue'] = df['Quantity'] * df['Price']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    customer_matrix = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalRevenue': 'sum',
        'StockCode': 'nunique'
    })

    customer_matrix.columns = ['Recency', 'Frequency', 'Monetary', 'Unique_Products']

    # -------- Feature Engineering --------
    first_purchase = df.groupby('Customer ID')['InvoiceDate'].min()
    total_days_active = (snapshot_date - first_purchase).dt.days
    total_days_active = total_days_active.reindex(customer_matrix.index)

    customer_matrix['Customer_Lifetime_Days'] = total_days_active

    customer_matrix['Expansion_Velocity'] = (
        customer_matrix['Unique_Products'] / (total_days_active + 1)
    )

    purchase_std = df.groupby('Customer ID')['InvoiceDate'].apply(
        lambda x: x.sort_values().diff().dt.days.std()
    )
    customer_matrix['Purchase_Consistency'] = purchase_std.reindex(customer_matrix.index).fillna(0)

    total_items = df.groupby('Customer ID')['Quantity'].sum()
    total_items = total_items.reindex(customer_matrix.index)
    customer_matrix['Items_Per_Order'] = total_items / customer_matrix['Frequency']

    customer_matrix['Purchase_Frequency_Rate'] = (
        customer_matrix['Frequency'] / (customer_matrix['Customer_Lifetime_Days'] + 1)
    )

    # -------- Outlier Handling --------
    customer_matrix['Monetary'] = customer_matrix['Monetary'].clip(upper=50000)
    customer_matrix['Items_Per_Order'] = customer_matrix['Items_Per_Order'].clip(upper=1000)
    customer_matrix['Expansion_Velocity'] = customer_matrix['Expansion_Velocity'].clip(upper=5)
    customer_matrix['Purchase_Consistency'] = customer_matrix['Purchase_Consistency'].clip(upper=60)

    # -------- Log Features --------
    customer_matrix['Log_Frequency'] = np.log1p(customer_matrix['Frequency'])
    customer_matrix['Log_Unique_Products'] = np.log1p(customer_matrix['Unique_Products'])

    # -------- Fill Missing --------
    customer_matrix = customer_matrix.fillna(0)

    return customer_matrix


# -------- EXECUTION --------
final_df = get_final_b2b_matrix(df)

output_dir = r"AI_Powered_CRM\backend\services\client lifetime value prediction\data"
os.makedirs(output_dir, exist_ok=True)

final_df.to_csv(os.path.join(output_dir, "cleaned_dataset.csv"))

# -------- PLOTS --------
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

for col in final_df.columns:
    plt.figure()
    final_df[col].hist()
    plt.title(col)
    plt.savefig(os.path.join(plot_dir, f"{col}.png"))
    plt.close()

# correlation heatmap
plt.figure(figsize=(10, 8))
corr = final_df.corr()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(plot_dir, "correlation.png"))
plt.close()

print("✓ Dataset + Plots Saved!")
print(final_df.head())