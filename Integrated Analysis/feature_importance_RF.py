import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("C:/Users/sooda/Climate Analysis Shimla/project/Integrated Analysis/integrated_data.csv")
df.columns = df.columns.str.strip()  # Remove whitespace from column names

# Print columns for verification
print("Available Columns:", df.columns.tolist())

# === Select relevant features only ===
features = ['Cropland %', 'Forest %', 'Urban %', 'LST', 'AQI']
target = 'Tourists footfall'

# Define X and y
X = df[features]
y = df[target]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data without shuffling to preserve temporal order
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Fit Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Get and print feature importances
importances = model.feature_importances_
print("\n=== Feature Importance ===")
for feat, imp in zip(features, importances):
    print(f"{feat}: {imp:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (Selected LULC + AQI, LST)")
plt.tight_layout()
plt.show()
