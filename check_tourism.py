import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla2_combined.csv")

# Select Relevant Columns
columns = ["year", "lst", "Total Tourists footfall"]
df = df[columns]

for lag in range(1, 4):  # Check 1 to 3 years lag
    df[f"Tourism_Lag_{lag}"] = df["Total Tourists footfall"].shift(lag)

lagged_corr = df.corr()
print(lagged_corr["lst"])

# Compute Correlation
correlation = df.corr(method='pearson')  # Pearson for linear relationship
spearman_corr = df.corr(method='spearman')  # Spearman for nonlinear

print("Pearson Correlation:\n", correlation)
print("Spearman Correlation:\n", spearman_corr)

# Visualize Correlation
plt.figure(figsize=(6, 4))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm")
plt.title("LST & Tourism Spearman Correlation Heatmap")
plt.show()
