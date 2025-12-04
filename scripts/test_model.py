import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

DATA_DIR = os.path.join("data_split", "test")
MODEL_PATH = os.path.join("models", "skin_disease_model.h5")

print("\n🔍 Loading model from:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

print("\n📂 Loading test dataset...")
test_ds = image_dataset_from_directory(
    DATA_DIR,
    image_size=(224, 224),
    batch_size=32
)

class_names = test_ds.class_names
print("✅ Classes:", class_names)

print("\n🧪 Evaluating model...")
loss, acc = model.evaluate(test_ds)
print(f"\n📊 Test Accuracy: {acc*100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")
