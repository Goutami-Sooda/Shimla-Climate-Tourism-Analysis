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

# Prepare LST Data for Forecasting
X_lst = df[["lst"]].values
y_lst = df[["Total Tourists footfall"]].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_lst_scaled = scaler_X.fit_transform(X_lst)
y_lst_scaled = scaler_y.fit_transform(y_lst)

# Reshape for GRU
X_lst_scaled = X_lst_scaled.reshape((X_lst_scaled.shape[0], 1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_lst_scaled, y_lst_scaled, test_size=0.2, random_state=42)

# Train LST GRU Model
model_lst = Sequential([
    GRU(64, activation='tanh', return_sequences=True, input_shape=(1, 1)),
    GRU(32, activation='tanh'),
    Dense(1)  # Predict future LST
])

model_lst.compile(optimizer='adam', loss='mse')
model_lst.fit(X_lst_scaled, y_lst_scaled, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Predict LST for Next 12 Months
future_dates_lst = pd.date_range(start="2023-01", periods=12, freq="MS")
future_X = np.array([[df["lst"].values[-1]]])  # Start with last known LST

predictions_lst = []
for _ in range(12):
    future_X_scaled = scaler_X.transform(future_X.reshape(1, -1)).reshape(1, 1, -1)
    predicted_scaled = model_lst.predict(future_X_scaled)
    predicted = scaler_y.inverse_transform(predicted_scaled)
    predictions_lst.append(predicted[0, 0])
    future_X = np.array([[predicted[0, 0]]])

# Plot Future LST
plt.figure(figsize=(10, 5))
plt.plot(future_dates_lst, predictions_lst, label="Predicted LST", color="red")
plt.legend()
plt.title("🚦 Forecasted LST for Crowd Management")
plt.xlabel("Date")
plt.ylabel("LST (°C)")
plt.show()
