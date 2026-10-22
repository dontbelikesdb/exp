
# Experiment 25: Semantic Segmentation using CNN
# Dataset: Oxford-IIIT Pet

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# 1. Load dataset
(ds_train, ds_test), info = tfds.load(
    "oxford_iiit_pet",
    split=["train", "test"],
    with_info=True
)

# 2. Preprocess images and masks
def preprocess(example):
    image = tf.image.resize(example["image"], (128, 128)) / 255.0

    mask = tf.image.resize(
        example["segmentation_mask"],
        (128, 128),
        method="nearest"
    )

    mask = tf.cast(mask > 1, tf.float32)
    return image, mask

train = (
    ds_train.map(preprocess)
    .shuffle(500)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

test = (
    ds_test.map(preprocess)
    .batch(16)
    .prefetch(tf.data.AUTOTUNE)
)

# 3. Build segmentation CNN
model = tf.keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.UpSampling2D(),
    layers.Conv2D(1, 1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train model
history = model.fit(
    train,
    epochs=3,
    validation_data=test
)

print("\nSegmentation model trained successfully!")

# 5. Display training results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.show()

# 6. Visualize segmentation results
for images, masks in test.take(1):
    predictions = model.predict(images, verbose=0)

    plt.figure(figsize=(12, 9))

    for i in range(min(3, len(images))):
        # Original image
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[i])
        plt.title("Original Image")
        plt.axis("off")

        # Actual segmentation mask
        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(tf.squeeze(masks[i]), cmap="gray", vmin=0, vmax=1)
        plt.title("Actual Mask")
        plt.axis("off")

        # Predicted segmentation mask
        predicted_mask = (predictions[i] > 0.5).astype("float32")

        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(predicted_mask.squeeze(), cmap="gray", vmin=0, vmax=1)
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

