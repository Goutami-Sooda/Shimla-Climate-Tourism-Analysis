import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("C:/Users/sooda/Climate Analysis Shimla/project/LULC Analysis/lulc_tourism_data.csv")
df.columns = df.columns.str.strip()

# Select only the LULC columns
lulc_cols = ['Bare land %', 'Cropland %', 'Forest %', 'Grassland %', 'Urban %', 'Water bodies %']
lulc_df = df[lulc_cols]

# Compute Kendall correlation matrix
corr_matrix = lulc_df.corr(method='kendall')

# Print correlation matrix
print("\n=== Kendall Correlation Matrix (LULC only) ===")
print(corr_matrix.round(2))

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Kendall Correlation Matrix - LULC Classes")
plt.tight_layout()
plt.show()

# Threshold for high correlation (absolute value)
threshold = 0.70

# Identify highly correlated pairs
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > threshold:
            print(f"Dropping '{col1}' due to high correlation with '{col2}' (r = {corr_val:.2f})")
            to_drop.add(col1)

# Retain only non-redundant features
selected_lulc_features = [col for col in lulc_cols if col not in to_drop]

print("\n=== Selected LULC Features After Correlation Filtering ===")
print(selected_lulc_features)
