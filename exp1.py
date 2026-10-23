import tensorflow as tf
from tensorflow.keras import layers, models

# Tensor operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print("Addition:", a + b)
print("Multiplication:", a * b)

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Single-layer perceptron
model1 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(10, activation='softmax')
])
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=3, verbose=0)
print("Perceptron accuracy:", model1.evaluate(x_test, y_test, verbose=0)[1])

# Multilayer perceptron
model2 = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=3, verbose=0)
print("MLP accuracy:", model2.evaluate(x_test, y_test, verbose=0)[1])