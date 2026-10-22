import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Load pretrained ResNet50
base = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False,
    pooling='avg'
)

# Extract image features
img = tf.keras.utils.load_img('image.jpg', target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.resnet50.preprocess_input(x)

features = base.predict(x, verbose=0)
print("Image features:", features.shape)

# Simple LSTM caption model
inp1 = layers.Input(shape=(2048,))
inp2 = layers.Input(shape=(10,))
emb = layers.Embedding(5000, 32)(inp2)
feat = layers.Dense(32, activation='relu')(inp1)
feat = layers.RepeatVector(10)(feat)

out = layers.Concatenate()([emb, feat])
out = layers.LSTM(64, return_sequences=True)(out)
out = layers.Dense(5000, activation='softmax')(out)

model = Model([inp1, inp2], out)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

print(model.summary())