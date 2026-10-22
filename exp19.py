import os

import certifi

# This Anaconda installation points OpenSSL to an expired system certificate
# bundle. Use certifi's maintained bundle for the TensorFlow Hub download.
os.environ["SSL_CERT_FILE"] = certifi.where()

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load pretrained object detection model
url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(url)

# Load an image named image.jpg
img = Image.open("image.jpg").convert("RGB").resize((320, 320))
x = np.array(img)

# Perform detection
result = detector(tf.constant(x[None, ...], dtype=tf.uint8))

boxes = result["detection_boxes"][0].numpy()
scores = result["detection_scores"][0].numpy()
classes = result["detection_classes"][0].numpy().astype(int)

# Display image
plt.imshow(x)
h, w = x.shape[:2]

for box, score, cls in zip(boxes, scores, classes):
    if score > 0.5:
        y1, x1, y2, x2 = box
        plt.gca().add_patch(
            plt.Rectangle((x1*w, y1*h),
                          (x2-x1)*w, (y2-y1)*h,
                          fill=False, edgecolor="red", linewidth=2)
        )
        plt.text(x1*w, y1*h, f"Class {cls}: {score:.2f}",
                 color="yellow")

plt.axis("off")
plt.show()
