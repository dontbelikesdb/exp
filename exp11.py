# Practical 11: Generative Adversarial Network (GAN)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Sequential

np.random.seed(42)
tf.random.set_seed(42)

# 1. Load MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(10000).batch(128, drop_remainder=True)

# 2. Generator
latent_dim = 100

generator = Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(256),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(512),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(784, activation="sigmoid")
])

# 3. Discriminator
discriminator = Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dropout(0.3),
    layers.Dense(256),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Dense(1, activation="sigmoid")
])

d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
bce = tf.keras.losses.BinaryCrossentropy()

# 4. Training step
@tf.function
def train_step(real_images):
    noise = tf.random.normal((128, latent_dim))

    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(tf.stop_gradient(fake_images), training=True)

        d_loss = (
            bce(tf.ones_like(real_output) * 0.9, real_output)
            + bce(tf.zeros_like(fake_output), fake_output)
        )

    d_gradients = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    noise = tf.random.normal((128, latent_dim))

    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_output = discriminator(fake_images, training=True)
        g_loss = bce(tf.ones_like(fake_output), fake_output)

    g_gradients = g_tape.gradient(
        g_loss, generator.trainable_variables
    )
    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )

    return d_loss, g_loss

# 5. Train GAN
for epoch in range(10):
    d_losses, g_losses = [], []

    for real_images in dataset:
        d_loss, g_loss = train_step(real_images)
        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

    print(
        f"Epoch {epoch + 1}/10 | "
        f"D Loss: {np.mean(d_losses):.4f} | "
        f"G Loss: {np.mean(g_losses):.4f}"
    )

# 6. Generate digits
noise = tf.random.normal((10, latent_dim))
images = generator(noise, training=False).numpy().reshape(-1, 28, 28)

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

plt.suptitle("GAN Generated MNIST Digits")
plt.tight_layout()
plt.show()

