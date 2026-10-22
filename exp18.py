import tensorflow as tf
from tensorflow.keras import layers

# Load CIFAR-10
(x, y), (xt, yt) = tf.keras.datasets.cifar10.load_data()

x = x.astype("float32") / 255
xt = xt.astype("float32") / 255

# Data augmentation
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

# CNN model
model = tf.keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    augment,
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x, y, epochs=3, batch_size=128, validation_split=0.1)
print("Test Accuracy:", model.evaluate(xt, yt, verbose=0)[1])



