import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam


# Load Data (Assuming 'df' contains 'lst' and 'Total Tourists footfall')
df = pd.read_csv("C:\\Users\\KRITI KANNAN\\Climate\\Shimla-Climate-Tourism-Analysis\\monthly_data_with_tourism.csv")
print(f"Initial dataset size: {df.shape}")
nan_count_before = df.isna().sum().sum()
print(f"Total NaN values before dropping: {nan_count_before}")

# Create Lagged Features for Tourism Footfall
df["Tourism_Lag_1"] = df["Total Tourists footfall"].shift(1)
df["Tourism_Lag_2"] = df["Total Tourists footfall"].shift(2)
df["Tourism_Lag_3"] = df["Total Tourists footfall"].shift(3)


# Drop rows where any column has NaN values
df.dropna(inplace=True)

# Debugging: Check NaN values in each column
nan_columns = df.columns[df.isna().any()].tolist()
if nan_columns:
    print("Columns with NaN values:", nan_columns)
    raise ValueError(f"NaN detected in columns: {nan_columns}")

if df.empty:
    raise ValueError("DataFrame is empty after shifting and dropping NaN values.")
print(f"final dataset size: {df.shape}")
print(df.tail())
# Check if there are enough rows after dropping NaNs
if df.empty:
    raise ValueError("No data available after dropping NaN values. Check your dataset or feature engineering.")
nan_count_after = df.isna().sum().sum()
print(f"Total NaN values after dropping: {nan_count_after}")

# Select Features (Lagged Tourism + Climate Variables)
features = ["lst", "satze", "sataz", "solze", "solaz", "Tourism_Lag_1", "Tourism_Lag_2", "Tourism_Lag_3"]
target = ["lst", "Total Tourists footfall"]  # Multi-output prediction

# Prepare Inputs and Outputs
X = df[features].values
y = df[target].values

# Check for NaNs again before scaling
if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
    print("NaN found in data before scaling! Replacing with column median...")
    df.fillna(df.median(), inplace=True)  # Fill missing values with median
    X = df[features].values
    y = df[target].values

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Verify Scaling Process
if np.isnan(X_scaled).sum() > 0 or np.isnan(y_scaled).sum() > 0:
    raise ValueError("NaN detected in dataset after scaling. Check for infinite values.")

# Reshape for GRU (Samples, Time Steps, Features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define GRU Model
model = Sequential([
    GRU(64, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[2])),
    GRU(32, activation='tanh'),
    Dense(2)  # Predicting both LST and Tourism Footfall
])

# Compile Model with a lower learning rate to avoid NaNs
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Make Predictions
y_pred = model.predict(X_test)

# Inverse Transform Predictions and Actual Values
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# Calculate RMSE for Both Targets
rmse_lst = np.sqrt(np.mean((y_test_inv[:, 0] - y_pred_inv[:, 0]) ** 2))
rmse_tourism = np.sqrt(np.mean((y_test_inv[:, 1] - y_pred_inv[:, 1]) ** 2))

print(f"RMSE for LST: {rmse_lst}")
print(f"RMSE for Tourism Footfall: {rmse_tourism}")