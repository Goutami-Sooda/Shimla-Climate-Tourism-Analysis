import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
df = pd.read_csv("C:\\Users\\KRITI KANNAN\\Climate\\Shimla-Climate-Tourism-Analysis\\monthly_data_with_tourism.csv")


# Apply Log Transformation to Tourism Footfall
df["Total Tourists footfall"] = np.log1p(df["Total Tourists footfall"])

# Create Multiple Lagged Features for Tourism Footfall (Up to 6 Months)
for lag in range(1, 7):
    df[f"Tourism_Lag_{lag}"] = df["Total Tourists footfall"].shift(lag)

# Drop Rows with NaN Values After Lagging
df.dropna(inplace=True)


# Select Features (LST + Past Tourism)
features = ["lst"] + [f"Tourism_Lag_{i}" for i in range(1, 7)]
target = ["Total Tourists footfall"]

X = df[features].values
y = df[target].values

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape for GRU
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define GRU Model
model = Sequential([
    GRU(64, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[2])),
    GRU(32, activation='tanh'),
    Dense(1)  # Predict tourism footfall
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Predict Future Demand
future_dates = pd.date_range(start="2030-01", periods=24, freq="MS")  # Predict next 2 years
predictions = model.predict(X_test)

# Inverse Transform Predictions
y_pred_inv = scaler_y.inverse_transform(predictions)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(future_dates, y_pred_inv[:24], label="Predicted Tourism Demand", color="blue")
plt.legend()
plt.title("📈 Predicted Tourism Demand (Next 2 Years)")
plt.xlabel("Date")
plt.ylabel("Tourist Footfall")
plt.show()
