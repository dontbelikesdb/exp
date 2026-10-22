import os

import certifi

# This Anaconda installation points OpenSSL to an expired system certificate
# bundle. Use certifi's maintained bundle for the dataset download.
os.environ["SSL_CERT_FILE"] = certifi.where()

import tensorflow as tf
from tensorflow.keras import layers

# Load the IMDB text dataset used by this binary-classification model.
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.imdb.load_data(num_words=5000)
)

x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train[:5000], maxlen=100
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test[:1000], maxlen=100
)
y_train, y_test = y_train[:5000], y_test[:1000]

# Transformer model
inp = layers.Input(shape=(100,))
x = layers.Embedding(5000, 64)(inp)

att = layers.MultiHeadAttention(
    num_heads=2, key_dim=32
)(x, x)

x = layers.Add()([x, att])
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inp, out)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=2, batch_size=64)
print("Test results:", model.evaluate(x_test, y_test))
