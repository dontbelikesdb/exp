import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Load MNIST
(x, _), (xt, _) = tf.keras.datasets.mnist.load_data()
x = x.astype("float32") / 255
xt = xt.astype("float32") / 255

# Add noise
xn = np.clip(x + np.random.normal(0, 0.3, x.shape), 0, 1)
xnt = np.clip(xt + np.random.normal(0, 0.3, xt.shape), 0, 1)

# Autoencoder
model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(784, activation="sigmoid"),
    layers.Reshape((28, 28, 1))
])

model.compile(optimizer="adam", loss="mse")
model.fit(xn[..., None], x[..., None],
          epochs=3, batch_size=128)

# Denoise images
pred = model.predict(xnt[:5, ..., None], verbose=0)

# Display results
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(3, 5, i+1)
    plt.imshow(xnt[i], cmap="gray")
    plt.axis("off")

    plt.subplot(3, 5, i+6)
    plt.imshow(pred[i].squeeze(), cmap="gray")
    plt.axis("off")

    plt.subplot(3, 5, i+11)
    plt.imshow(xt[i], cmap="gray")
    plt.axis("off")

plt.show()