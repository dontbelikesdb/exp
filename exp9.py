import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Sample user-product interactions
users = np.array([0, 0, 1, 1, 2, 2])
items = np.array([0, 1, 1, 2, 0, 2])
ratings = np.array([1, 1, 1, 1, 1, 1], dtype='float32')

# Embedding model
u = layers.Input(shape=(1,))
i = layers.Input(shape=(1,))

ue = layers.Flatten()(layers.Embedding(3, 8)(u))
ie = layers.Flatten()(layers.Embedding(3, 8)(i))

x = layers.Concatenate()([ue, ie])
x = layers.Dense(16, activation='relu')(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = Model([u, i], out)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit([users, items], ratings, epochs=20, verbose=0)

# Recommend products for user 0
scores = model.predict([
    np.zeros(3, dtype='int32'),
    np.arange(3)
], verbose=0).flatten()

print("Recommended product:", np.argmax(scores))
print("Product scores:", scores)