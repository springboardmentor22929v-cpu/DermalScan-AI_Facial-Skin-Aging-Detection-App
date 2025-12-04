import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

BASE_DIR = "data_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("📂 Loading datasets...")

train_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(BASE_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

NUM_CLASSES = len(train_ds.class_names)
print(f"Detected classes: {train_ds.class_names}")

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("🔧 Building model...")

base_model = keras.applications.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🚀 Training model...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

print("💾 Saving model to models/skin_disease_model.h5")

os.makedirs("models", exist_ok=True)
model.save("models/skin_disease_model.h5")

print("🎉 Training complete!")
