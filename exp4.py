import tensorflow as tf
from tensorflow.keras import layers, models

# Load images
train = tf.keras.utils.image_dataset_from_directory(
    "dataset/train", image_size=(160, 160), batch_size=32
)
val = tf.keras.utils.image_dataset_from_directory(
    "dataset/validation", image_size=(160, 160), batch_size=32
)

# Pretrained model
base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False,
    weights='imagenet'
)
base.trainable = False

# Build classifier
model = models.Sequential([
    layers.Input(shape=(160, 160, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Feature extraction
model.fit(train, validation_data=val, epochs=3)
print("Feature extraction accuracy:",
      model.evaluate(val, verbose=0)[1])

# Fine-tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, validation_data=val, epochs=2)
print("Fine-tuning accuracy:", model.evaluate(val, verbose=0)[1])