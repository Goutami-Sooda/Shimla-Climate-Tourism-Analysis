import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("lulc_tourism_data.csv")

# Define the lags to analyze
max_lag = 4
lags = range(0, max_lag + 1)

# Initialize a list to store lagged correlations
correlations = []

# Loop through each lag
for lag in lags:
    # Create a copy of the original dataset for each lag
    lagged_data = data.copy()

    # Shift the tourist footfall data for the current lag
    lagged_data[f'Tourism Lag {lag}'] = data['Tourists footfall'].shift(-lag) # '-' for downward lag, '+' for upward

    # Print the data after lagging (before dropping NaN values)
    print("Data after {lag} lagging:")
    print(lagged_data)

    # Drop rows with NaN values caused by lagging
    lagged_data = lagged_data.dropna()

    # Print the cleaned data after dropping NaN values
    print("\nData after dropping NaN values:")
    print(lagged_data)

    # Compute the correlation for the current lag
    lag_corr = {
        'Lag': lag,
        'Bare land': lagged_data['Bare land %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall'),
        'Cropland': lagged_data['Cropland %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall'),
        'Forest': lagged_data['Forest %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall'),
        'Grassland': lagged_data['Grassland %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall'),
        'Urban': lagged_data['Urban %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall'),
        'Water bodies': lagged_data['Water bodies %'].corr(lagged_data[f'Tourism Lag {lag}'], method='kendall')
    }

    correlations.append(lag_corr)

# Convert the list of correlations to a DataFrame
correlations_df = pd.DataFrame(correlations)

# Print the correlations
print(correlations_df)

# Plot the lagged correlations
plt.figure(figsize=(10, 6))
for col in correlations_df.columns[1:]:
    plt.plot(correlations_df['Lag'], correlations_df[col], label=col, marker='o')

plt.title("Lagged Correlation Analysis")
plt.xlabel("Lag (Years)")
plt.ylabel("Correlation")
plt.legend(title="LULC Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
