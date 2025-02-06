import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import chardet

# Load dataset with encoding detection
def load_data(file_path, column_name):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    encoding = result['encoding']
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        fallback_encodings = ['latin1', 'iso-8859-1', 'cp1252']
        for enc in fallback_encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"File successfully loaded with {enc} encoding.")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Failed to decode the file.")
    
    return df[column_name].values.reshape(-1, 1)

# Wavelet decomposition
def wavelet_decompose(series, wavelet='db4', level=3):
    coeffs = pywt.wavedec(series.flatten(), wavelet, level=level)
    return coeffs

# Reconstruct the signal
def wavelet_reconstruct(coeffs, wavelet='db4'):
    return pywt.waverec(coeffs, wavelet)

# Prepare sequences for Decision Tree
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

# Train Decision Tree and make predictions
def train_and_predict(file_path, column_name, time_steps=10):
    # Load and normalize data
    data = load_data(file_path, column_name)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Wavelet decomposition
    coeffs = wavelet_decompose(data_scaled)
    reconstructed_series = [wavelet_reconstruct([c if i == j else np.zeros_like(c) for j, c in enumerate(coeffs)]) for i in range(len(coeffs))]

    predictions = []
    actual_values = []

    for series in reconstructed_series:
        X, y = create_sequences(series.reshape(-1, 1), time_steps)
        train_X, train_y = X[:-10], y[:-10]  # Train set
        test_X, test_y = X[-10:], y[-10:]    # Test set

        # Train Decision Tree model
        dt_reg = DecisionTreeRegressor(random_state=0)
        dt_reg.fit(train_X.reshape(train_X.shape[0], -1), train_y)

        # Make predictions
        pred = dt_reg.predict(test_X.reshape(test_X.shape[0], -1))
        predictions.append(pred)
        actual_values.append(test_y.flatten())

    # Convert predictions to NumPy arrays
    predictions = np.mean(np.array(predictions), axis=0)
    actual_values = np.mean(np.array(actual_values), axis=0)

    # Inverse transform predictions and actual values to get real AQI values
    predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual_values_actual = scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()

    # Compute evaluation metrics
    r2 = r2_score(actual_values, predictions)
    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))

    return predictions, predictions_actual, actual_values_actual, r2, mae, rmse

# Example Usage
file_path = 'AQI_data.csv'  # Replace with actual file
column_name = 'AQI'         # Replace with actual column name

predictions_normalized, predictions_actual, actual_values_actual, r2, mae, rmse = train_and_predict(file_path, column_name)

# Print Results
print("\n--- Predicted AQI Values (Normalized) ---")
print(predictions_normalized)

print("\n--- Predicted AQI Values (Actual) ---")
print(predictions_actual)

print("\n--- Evaluation Metrics ---")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse:.4f}")

"""OUTPUT:
--- Predicted AQI Values (Normalized) ---
[0.12313005 0.10799316 0.21830973 0.23881208 0.08425666 0.06995311
 0.1024299  0.17604549 0.11931516 0.12721138]

--- Predicted AQI Values (Actual) ---
[30.04061419 28.53582436 39.50262411 41.54080597 26.17612948 24.75418453
 27.98276931 35.30104834 29.66136835 30.44634776]

--- Evaluation Metrics ---
R² Score: 0.9401
Mean Absolute Error (MAE): 0.0111
Root Mean Square Error (RMSE): 0.0138
"""