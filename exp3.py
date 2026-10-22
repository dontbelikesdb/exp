import os

import certifi

# This Anaconda installation points OpenSSL to an expired system certificate
# bundle. Use certifi's maintained bundle for the CIFAR-10 HTTPS download.
os.environ["SSL_CERT_FILE"] = certifi.where()

import tensorflow as tf
from tensorflow.keras import layers, models

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build CNN
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)

# Save and reload
model.save("cnn_model.keras")
model = tf.keras.models.load_model("cnn_model.keras")

print("Test accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])
