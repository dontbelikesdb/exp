import tensorflow as tf
from tensorflow.keras import layers, models
import time

# Load IMDB dataset
vocab = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.imdb.load_data(num_words=vocab)
)

x_train = tf.keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.utils.pad_sequences(x_test, maxlen=maxlen)

# Compare RNN, LSTM and GRU
for name, layer in [
    ("SimpleRNN", layers.SimpleRNN(32)),
    ("LSTM", layers.LSTM(32)),
    ("GRU", layers.GRU(32))
]:
    model = models.Sequential([
        layers.Embedding(vocab, 32),
        layer,
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    start = time.time()
    model.fit(x_train, y_train, epochs=2, batch_size=64, verbose=0)
    elapsed = time.time() - start

    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    print(name, "Accuracy:", accuracy, "Time:", round(elapsed, 2), "seconds")