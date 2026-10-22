import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile
import os

# Download Jena Climate dataset
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = tf.keras.utils.get_file(
    "jena_climate.zip", origin=url
)

folder = os.path.dirname(zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(folder)

df = pd.read_csv(os.path.join(folder, "jena_climate_2009_2016.csv"))
print("Dataset shape:", df.shape)

# Use temperature readings (every hour)
data = df["T (degC)"].values[5::6]

# Normalize data
mean, std = data.mean(), data.std()
data = (data - mean) / std

# Create sequences of 24 hours to predict the next hour
X, y = [], []
for i in range(len(data) - 25):
    X.append(data[i:i+24])
    y.append(data[i+24])

X = np.array(X).reshape(-1, 24, 1)
y = np.array(y)

# Split chronologically: 80% training, 20% testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 1)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=5, batch_size=128)

# Evaluate model
loss = model.evaluate(X_test, y_test, verbose=0)
print("Test MSE:", loss)

# Predict temperatures
pred = model.predict(X_test[:100], verbose=0)

# Convert values back to Celsius
actual = y_test[:100] * std + mean
pred = pred.flatten() * std + mean

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(actual, label="Actual Temperature")
plt.plot(pred, label="Predicted Temperature")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Jena Climate Temperature Prediction using LSTM")
plt.legend()
plt.show()