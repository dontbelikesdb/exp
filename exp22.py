import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# English and French sentence pairs
eng = [
    "hello", "thank you", "good morning", "how are you",
    "i am fine", "good night", "see you", "i love you",
    "thank you very much", "good evening"
]

fra = [
    "bonjour", "merci", "bonjour", "comment allez vous",
    "je vais bien", "bonne nuit", "au revoir", "je taime",
    "merci beaucoup", "bonsoir"
]

# Add start and end tokens to French sentences
fra = ["start " + s + " end" for s in fra]

# Tokenization
src = layers.TextVectorization(output_mode="int")
tgt = layers.TextVectorization(output_mode="int")

src.adapt(eng)
tgt.adapt(fra)

X = src(np.array(eng))
Y = tgt(np.array(fra))

src_vocab = len(src.get_vocabulary())
tgt_vocab = len(tgt.get_vocabulary())

# Encoder
enc_input = layers.Input(shape=(None,))
enc_emb = layers.Embedding(src_vocab, 64)(enc_input)

enc_output, state_h, state_c = layers.LSTM(
    64, return_sequences=True, return_state=True
)(enc_emb)

# Decoder
dec_input = layers.Input(shape=(None,))
dec_emb = layers.Embedding(tgt_vocab, 64)(dec_input)

dec_output = layers.LSTM(
    64, return_sequences=True
)(dec_emb, initial_state=[state_h, state_c])

# Attention mechanism
attention = layers.Attention()([dec_output, enc_output])

combined = layers.Concatenate()([dec_output, attention])
output = layers.Dense(tgt_vocab, activation="softmax")(combined)

# Build model
model = tf.keras.Model([enc_input, dec_input], output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    [X, Y[:, :-1]],
    Y[:, 1:, None],
    epochs=300,
    verbose=0
)

print("Model training completed!")

# Translation function
def translate(sentence):
    source = src(np.array([sentence]))

    # Start with the French start token
    start_id = tgt.get_vocabulary().index("start")
    end_id = tgt.get_vocabulary().index("end")
    target = [start_id]

    for _ in range(10):
        decoder_input = np.array([target])

        predictions = model.predict(
            [source, decoder_input], verbose=0
        )

        next_id = int(np.argmax(predictions[0, -1, :]))

        if next_id == end_id or next_id == 0:
            break

        target.append(next_id)

    vocabulary = tgt.get_vocabulary()
    words = [vocabulary[i] for i in target[1:]]
    return " ".join(words)

# Test translations
for sentence in ["hello", "thank you", "good night"]:
    print("English:", sentence)
    print("French:", translate(sentence))
    print()