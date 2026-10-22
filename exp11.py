import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

# Load MNIST dataset
(x, _), (_, _) = tf.keras.datasets.mnist.load_data()
x = (x.astype("float32") - 127.5) / 127.5
x = x.reshape(-1, 784)

# Generator
G = tf.keras.Sequential([
    layers.Input(shape=(100,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(784, activation="tanh")
])

# Discriminator
D = tf.keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

D.compile(optimizer="adam", loss="binary_crossentropy")

# GAN
D.trainable = False
z = layers.Input(shape=(100,))
gan = Model(z, D(G(z)))
gan.compile(optimizer="adam", loss="binary_crossentropy")

# Training
for epoch in range(50):
    for i in range(0, len(x), 128):
        real = x[i:i+128]
        n = len(real)

        noise = np.random.normal(0, 1, (n, 100))
        fake = G.predict(noise, verbose=0)

        D.trainable = True
        D.train_on_batch(real, np.ones((n, 1)))
        D.train_on_batch(fake, np.zeros((n, 1)))

        noise = np.random.normal(0, 1, (n, 100))
        D.trainable = False
        gan.train_on_batch(noise, np.ones((n, 1)))

    if (epoch + 1) % 10 == 0:
        print("Epoch:", epoch + 1)

# Generate images
noise = np.random.normal(0, 1, (10, 100))
images = G.predict(noise, verbose=0)
images = (images + 1) / 2

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.show()