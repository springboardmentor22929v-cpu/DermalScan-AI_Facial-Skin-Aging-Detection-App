# File: scripts/train_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import os
import datetime
import matplotlib.pyplot as plt
from prepare_dataset import train_ds, val_ds, test_ds, IMG_SIZE, class_names, num_classes

# --- Configuration ---
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models"
LOGS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\logs"
HISTORY_PLOT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\training_plots"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(HISTORY_PLOT_DIR, exist_ok=True)

print(f"Training with {num_classes} classes: {class_names}")

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- Base Model ---
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model initially
base_model.trainable = False

# --- Build Model ---
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)  # Apply augmentation
x = layers.Rescaling(1./255)(x)  # Normalize pixels
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs, name="dermalscan_model")
model.summary()

# --- Callbacks ---
log_dir = os.path.join(LOGS_DIR, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_filepath = os.path.join(MODELS_DIR, 'best_model.h5')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

callbacks = [tensorboard_callback, model_checkpoint_callback, early_stopping_callback]

# --- Phase 1: Train Head ---
print("\n--- Phase 1: Training head ---")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

initial_epochs = 10
history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

# --- Phase 2: Fine-tuning ---
print("\n--- Phase 2: Fine-tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Freeze all except last 20 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=initial_epochs,
    validation_data=val_ds,
    callbacks=callbacks
)

# --- Plot History ---
def plot_history(history_dict, phase_name, filename_suffix=""):
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{phase_name} Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{phase_name} Loss')

    plt.savefig(os.path.join(HISTORY_PLOT_DIR, f"{phase_name.replace(' ', '_').lower()}{filename_suffix}.png"))
    plt.show()

# Combine histories
combined_history = {}
for key in history.history:
    combined_history[key] = history.history[key] + history_fine.history[key]

plot_history(combined_history, "Full Training")

# --- Save final model ---
final_model_path = os.path.join(MODELS_DIR, "dermalscan_model_final.h5")
model.save(final_model_path)
print(f"✅ Final model saved: {final_model_path}")

# --- Evaluate ---
print("\nEvaluating best model on test set...")
best_model = tf.keras.models.load_model(checkpoint_filepath)
test_loss, test_accuracy = best_model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
