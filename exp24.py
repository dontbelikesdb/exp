
# Practical 24: Credit Card Fraud Detection using Autoencoder

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
tf.random.set_seed(42)

# 1. Load dataset
df = pd.read_csv("creditcard.csv")
X = df.drop(columns=["Class", "Time"])
y = df["Class"].values

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scale features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# 4. Train only on legitimate transactions
X_normal = X_train[y_train == 0]

# 5. Build autoencoder
inputs = layers.Input(shape=(X_normal.shape[1],))
x = layers.Dense(16, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(X_normal.shape[1])(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse")

# 6. Train model
model.fit(
    X_normal, X_normal,
    epochs=10, batch_size=256,
    validation_split=0.1, verbose=1
)

# 7. Calculate reconstruction error
train_recon = model.predict(X_normal, verbose=0)
train_error = np.mean((X_normal - train_recon) ** 2, axis=1)
threshold = np.percentile(train_error, 95)

test_recon = model.predict(X_test, verbose=0)
test_error = np.mean((X_test - test_recon) ** 2, axis=1)

# 8. Detect fraud
y_pred = (test_error > threshold).astype(int)

# 9. Evaluate model
print("\nAnomaly Threshold:", round(threshold, 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(
          y_test, y_pred,
          target_names=["Normal", "Fraud"],
          zero_division=0
      ))
