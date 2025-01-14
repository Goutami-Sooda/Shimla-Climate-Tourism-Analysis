import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from CSV file
# Replace 'lulc_tourism_data.csv' with the path to your actual CSV file
df = pd.read_csv('lulc_tourism_data.csv')

# Ensure the 'Year' column is not used in the correlation
df = df.drop(columns=["Year"])  # Drop 'Year' if present

# Pearson correlation matrix
correlation_matrix = df.corr()

# Plot heat map
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)             
plt.title("Heat Map of Pearson Correlation Between LULC Classes and Tourist Footfall")
plt.tight_layout()  # Automatically adjusts layout to fit everything
plt.show()

# Spearman correlation
spearman_corr = df.corr(method='spearman')

# Plot heat map
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)   
plt.title("Heat Map of Spearman Correlation Between LULC Classes and Tourist Footfall")
plt.tight_layout()
plt.show()

# Kendall correlation
kendall_corr = df.corr(method='kendall')

# Plot heat map
plt.figure(figsize=(8, 6))
sns.heatmap(kendall_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)              
plt.title("Heat Map of Kendall Correlation Between LULC Classes and Tourist Footfall")
plt.tight_layout()
plt.show()