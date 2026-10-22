import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Load dataset
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

x_train = x_train / 255.0
x_test = x_test / 255.0

# Build MLP
def build_model(activation):
    model = Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation=activation),
        layers.Dense(64, activation=activation),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Compare activations
for activation in ["sigmoid", "tanh", "relu"]:
    model = build_model(activation)
    model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=128,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(activation, "Accuracy:", round(accuracy, 4))