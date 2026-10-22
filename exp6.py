import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Load data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Autoencoder
inp = layers.Input(shape=(784,))
encoded = layers.Dense(32, activation='relu')(inp)
decoded = layers.Dense(784, activation='sigmoid')(encoded)
model = Model(inp, decoded)

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=5, batch_size=256, verbose=1)

# Reconstruction
output = model.predict(x_test[:5])
print("Reconstruction error:", model.evaluate(x_test, x_test, verbose=0))

# Display original and reconstructed images
plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(output[0].reshape(28, 28), cmap='gray')
plt.title("Reconstructed")
plt.show()