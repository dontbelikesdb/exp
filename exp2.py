import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
def build_model(reg=None, drop=False):
    model = tf.keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=reg),
        layers.Dropout(0.3) if drop else layers.Activation('linear'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Compare models
for name, reg, drop in [
    ("Normal", None, False),
    ("L1", regularizers.l1(0.001), False),
    ("L2", regularizers.l2(0.001), False),
    ("Dropout", None, True)
]:
    model = build_model(reg, drop)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(name, "MSE:", loss, "MAE:", mae)