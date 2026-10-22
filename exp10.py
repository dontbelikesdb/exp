import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Sample text data
texts = [
    "I love this product",
    "This is very bad",
    "Amazing experience",
    "I hate this"
]
labels = np.array([1, 0, 1, 0], dtype=np.float32)

# Text vectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000, output_sequence_length=10
)
vectorizer.adapt(texts)
# Keras 3.15's data adapter expects NumPy arrays here rather than an
# EagerTensor returned directly by TextVectorization.
X = vectorizer(tf.constant(texts)).numpy()

# LSTM sentiment model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, labels, epochs=20, verbose=0)

# Predict sentiment
print("Sentiment scores:", model.predict(X, verbose=0).flatten())

# User interaction graph
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D')])

sentiment = {'A': 1, 'B': -1, 'C': 1, 'D': -1}

nx.draw(G, with_labels=True, node_color=[
    'lightgreen' if sentiment[n] == 1 else 'salmon' for n in G.nodes()
])
plt.title("User Sentiment Network")
plt.show()
