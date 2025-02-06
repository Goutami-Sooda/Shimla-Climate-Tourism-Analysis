import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the dataset
file_path = "lulc_tourism_data.csv"
df = pd.read_csv(file_path)
print(df.head())

# Select relevant features
features = ["Bare land %", "Cropland %", "Forest %", "Grassland %", "Urban %", "Water bodies %", "Tourists footfall"]

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Convert scaled data back to a DataFrame for easy handling
scaled_df = pd.DataFrame(scaled_data, columns=features)

# Initialize and fit Isolation Forest with adjusted parameters
model = IsolationForest(
    n_estimators=100,  # Experiment with increasing the number of estimators
    contamination=0.08,  # Reduce contamination to minimize false positives
    random_state=42
)
df['Anomaly'] = model.fit_predict(scaled_data)  # -1 for anomalies, 1 for normal points
df['Anomaly Score'] = model.decision_function(scaled_data)  # Anomaly score

# Display anomalies
anomalies = df[df['Anomaly'] == -1]
print("Anomalies Detected:")
print(anomalies)

# Save the dataset with anomaly labels
output_file = "lulc_tourism_anomalies.csv"
df.to_csv(output_file, index=False)
print(f"Anomaly data saved to {output_file}")

# Remove 2009 data from plotting
df_plot = df[df['Year'] != 2009]

# Plot tourist footfall with anomalies highlighted
plt.figure(figsize=(10, 6))
plt.plot(df_plot['Year'], df_plot['Tourists footfall'], label='Tourists Footfall', marker='o')
plt.scatter(anomalies[anomalies['Year'] != 2009]['Year'], anomalies[anomalies['Year'] != 2009]['Tourists footfall'], color='red', label='Anomalies', s=100)
plt.xlabel('Year')
plt.ylabel('Tourists Footfall')
plt.title('Tourists Footfall and LULC with Anomalies Highlighted')
plt.legend()
plt.grid()
plt.show()

# Plot anomaly scores for all years
plt.figure(figsize=(10, 6))
plt.plot(df_plot['Year'], df_plot['Anomaly Score'], label='Anomaly Score', marker='o')
plt.axhline(y=0, color='red', linestyle='--', label='Threshold')
plt.xlabel('Year')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Scores Across Years')
plt.legend()
plt.grid()
plt.show()
