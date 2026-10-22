import numpy as np
import tensorflow as tf
import urllib.request

# Download Shakespeare text
url = "https://www.gutenberg.org/files/100/100-0.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")[:100000].lower()

# Character encoding
chars = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for c, i in char_to_id.items()}
data = np.array([char_to_id[c] for c in text])

# Create sequences
seq_len = 40
X, y = [], []

for i in range(0, len(data) - seq_len, 3):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 32),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X, y, epochs=10, batch_size=128)

# Generate text
seed = text[:40]

for _ in range(300):
    seq = np.array([[char_to_id[c] for c in seed[-seq_len:]]])
    probs = model.predict(seq, verbose=0)[0]
    next_id = np.random.choice(len(chars), p=probs)
    seed += id_to_char[next_id]

print("\nGenerated Text:\n", seed[-300:])