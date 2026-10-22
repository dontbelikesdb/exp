import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Training sentences
sentences = [
    ["John", "lives", "in", "London"],
    ["Mary", "works", "at", "Google"],
    ["Alice", "visits", "Paris"],
    ["Bob", "works", "in", "India"]
]

# Tags: O=Other, PER=Person, LOC=Location, ORG=Organization
tags = [
    ["PER", "O", "O", "LOC"],
    ["PER", "O", "O", "ORG"],
    ["PER", "O", "LOC"],
    ["PER", "O", "O", "LOC"]
]

words = sorted(set(w for s in sentences for w in s))
labels = sorted(set(t for row in tags for t in row))

word2id = {w: i+1 for i, w in enumerate(words)}
tag2id = {t: i for i, t in enumerate(labels)}
id2tag = {i: t for t, i in tag2id.items()}

X = pad_sequences([[word2id[w] for w in s] for s in sentences],
                  padding="post")
Y = pad_sequences([[tag2id[t] for t in row] for row in tags],
                  padding="post")

# BiLSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word2id)+1, 16, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Dense(len(labels), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X, Y, epochs=100, verbose=0)

# Predict entities
sentence = ["John", "lives", "in", "London"]
seq = pad_sequences([[word2id.get(w, 0) for w in sentence]],
                    maxlen=X.shape[1], padding="post")
pred = model.predict(seq, verbose=0).argmax(axis=-1)[0]

for word, tag in zip(sentence, pred):
    print(word, ":", id2tag[tag])