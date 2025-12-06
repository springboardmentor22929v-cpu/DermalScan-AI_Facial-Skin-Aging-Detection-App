# File: scripts/fine_tune_model.py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import datetime
import matplotlib.pyplot as plt
from prepare_dataset import train_ds, val_ds, test_ds, class_names, num_classes

# ==============================
# Paths
# ==============================
BASE_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project"
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

best_model_path = os.path.join(MODELS_DIR, "efficientnetb0_skin_model_best.h5")
fine_tuned_model_path = os.path.join(MODELS_DIR, "efficientnetb0_skin_model_finetuned.h5")

# ==============================
# Load model
# ==============================
print("🔹 Loading best model for further fine-tuning...")
model = tf.keras.models.load_model(best_model_path)

# Get base EfficientNet model
base_model = model.get_layer("efficientnetb0")

# Unfreeze last 80 layers for deep fine-tuning
for layer in base_model.layers[:-80]:
    layer.trainable = False
for layer in base_model.layers[-80:]:
    layer.trainable = True

model.summary()

# ==============================
# Compile (IMPORTANT FIX)
# ==============================
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",     # <-- FIXED!!
    metrics=["accuracy"]
)

# ==============================
# Callbacks
# ==============================
log_dir = os.path.join(LOGS_DIR, "finetune", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=fine_tuned_model_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
]

# ==============================
# Class weights (optional)
# ==============================
class_weights = {0: 1.5, 1: 1.5, 2: 1.5, 3: 0.8}

# ==============================
# Fine-tune model
# ==============================
fine_tune_epochs = 40
print("\n🚀 Starting fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==============================
# Plot
# ==============================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Fine-tuning Accuracy")
plt.legend(["Train", "Val"])

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Fine-tuning Loss")
plt.legend(["Train", "Val"])

plt.show()

# ==============================
# Evaluate on test set
# ==============================
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Fine-tuning complete!")
print(f"📌 Test Accuracy: {test_acc*100:.2f}%")
print(f"📌 Model saved to: {fine_tuned_model_path}")
