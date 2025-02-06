import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load Dataset
df = pd.read_csv("C:\\Users\\KRITI KANNAN\\Climate\\Shimla-Climate-Tourism-Analysis\\monthly_data_with_tourism.csv")

# Apply Log Transformation to Tourism Footfall
df["Total Tourists footfall"] = np.log1p(df["Total Tourists footfall"])

# Create Multiple Lagged Features for Tourism Footfall (Up to 6 Months)
for lag in range(1, 7):
    df[f"Tourism_Lag_{lag}"] = df["Total Tourists footfall"].shift(lag)

# Drop Rows with NaN Values After Lagging
df.dropna(inplace=True)

# Feature Selection
features = ["lst", "satze", "sataz", "solze", "solaz"] + [f"Tourism_Lag_{i}" for i in range(1, 7)]
target = ["lst", "Total Tourists footfall"]

# Prepare Inputs and Outputs
X = df[features].values
y = df[target].values

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape for GRU (Samples, Time Steps, Features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define Improved GRU Model
model = Sequential([
    GRU(128, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    GRU(64, activation='tanh'),
    Dropout(0.2),
    Dense(2)  # Predict both LST & Tourism Footfall
])

# Compile Model with Lower Learning Rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# Make Predictions
y_pred = model.predict(X_test)

# Inverse Transform Predictions and Actual Values
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# Reverse Log Transformation for Tourism Footfall
y_pred_inv[:, 1] = np.expm1(y_pred_inv[:, 1])  # Convert back to original scale
y_test_inv[:, 1] = np.expm1(y_test_inv[:, 1])

# Calculate RMSE for Both Targets
rmse_lst = np.sqrt(np.mean((y_test_inv[:, 0] - y_pred_inv[:, 0]) ** 2))
rmse_tourism = np.sqrt(np.mean((y_test_inv[:, 1] - y_pred_inv[:, 1]) ** 2))

print(f"✅ RMSE for LST: {rmse_lst}")
print(f"✅ RMSE for Tourism Footfall: {rmse_tourism}")

# -------------------- PREDICTION FOR 2030 -------------------- #

# Generate Future Dates (Assuming Bi-Monthly Data)
future_dates = pd.date_range(start='2021-01', periods=120, freq='MS')  # 10 years till 2030

# Initialize Last Known Data for Prediction
last_known_data = df.iloc[-6:].copy()  # Last 6 months for lagged inputs

predictions_2030 = []

for date in future_dates:
    # Prepare Input for Prediction
    last_input = last_known_data[features].values[-1].reshape(1, 1, -1)  # Reshape for GRU
    last_input_scaled = scaler_X.transform(last_input.reshape(1, -1)).reshape(1, 1, -1)

    # Predict Next Month
    predicted_scaled = model.predict(last_input_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled)[0]

    # Reverse Log Transformation for Footfall
    predicted[1] = np.expm1(predicted[1])

    # Store Prediction
    predictions_2030.append([date, predicted[0], predicted[1]])  # Date, LST, Footfall

    # Update last_known_data with the new predicted values
    new_row = last_known_data.iloc[-1].copy()  # Copy last row
    new_row["lst"] = predicted[0]  # Update LST
    new_row["Total Tourists footfall"] = np.log1p(predicted[1])  # Update log-transformed footfall

    # Shift Lagged Values
    for lag in range(6, 1, -1):
        new_row[f"Tourism_Lag_{lag}"] = last_known_data[f"Tourism_Lag_{lag-1}"].iloc[-1]
    new_row["Tourism_Lag_1"] = np.log1p(predicted[1])  # Latest predicted footfall

    # Append new row to last_known_data
    last_known_data = pd.concat([last_known_data, new_row.to_frame().T]).reset_index(drop=True)

# Convert Predictions to DataFrame
predictions_df = pd.DataFrame(predictions_2030, columns=["Date", "Predicted_LST", "Predicted_Tourism_Footfall"])
predictions_df.to_csv("Predicted_2030_LST_Tourism.csv", index=False)

print("\n✅ Predictions for 2030 saved to 'Predicted_2030_LST_Tourism.csv'")
print(predictions_df.head(10))  # Display first 10 rows of predictions
