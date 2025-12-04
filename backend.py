import tensorflow as tf
import numpy as np
import os

MODEL_PATH = os.path.join("models", "skin_disease_model.h5")

# Load model only once
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['clear skin', 'dark spots', 'puffy eyes', 'wrinkles']


def predict_image(image):
    """
    Takes numpy image array and returns (label, confidence).
    """

    # Resize + normalize
    img = tf.image.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    idx = np.argmax(preds[0])

    label = class_names[idx]
    confidence = float(preds[0][idx] * 100)

    return label, confidence
