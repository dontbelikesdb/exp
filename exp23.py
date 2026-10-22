import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# Load MNIST
(x, _), (_, _) = tf.keras.datasets.mnist.load_data()
x = x.astype("float32") / 255
x = x.reshape(-1, 784)

latent = 2

# Encoder
inp = layers.Input(shape=(784,))
h = layers.Dense(256, activation="relu")(inp)
mu = layers.Dense(latent)(h)
log_var = layers.Dense(latent)(h)

def sample(args):
    mu, log_var = args
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps

z = layers.Lambda(sample)([mu, log_var])
encoder = tf.keras.Model(inp, [mu, log_var, z])

# Decoder
zi = layers.Input(shape=(latent,))
d = layers.Dense(256, activation="relu")(zi)
out = layers.Dense(784, activation="sigmoid")(d)
decoder = tf.keras.Model(zi, out)

# VAE
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        # Assigning these as model attributes registers their trainable
        # variables with the subclassed VAE.
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, lv, z = self.encoder(data)
            recon = self.decoder(z)
            rec_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, recon), axis=-1)
            )
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1)
            )
            loss = rec_loss + kl
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer="adam")
vae.fit(x, epochs=5, batch_size=128)

# Generate new digits
samples = decoder.predict(np.random.normal(size=(10, latent)), verbose=0)

plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(samples[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
plt.show()
