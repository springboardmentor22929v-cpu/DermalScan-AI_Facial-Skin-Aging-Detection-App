# File: scripts/train_model.py 
 
import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.optimizers import Adam 
import os 
import datetime # For TensorBoard logs 
import matplotlib.pyplot as plt # For plotting history 
 
# Import the dataset variables directly from prepare_dataset.py 
# Data comes in already augmented and normalized 
from prepare_dataset import train_ds, val_ds, test_ds, IMG_SIZE, BATCH_SIZE, class_names, num_classes 
 
# --- Configuration --- 
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models" 
LOGS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\logs" 
HISTORY_PLOT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\training_plots" 
 
os.makedirs(MODELS_DIR, exist_ok=True) 
os.makedirs(LOGS_DIR, exist_ok=True) 
os.makedirs(HISTORY_PLOT_DIR, exist_ok=True) 
 
print(f"Training will use {num_classes} classes: {class_names}") 
 
# --- Model Definition --- 
# Path to your manually downloaded weights file (kept for reference, but won't be directly loaded here) 
# LOCAL_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_base_notop.weights.h5') 

# --- NEW ROBUST APPROACH: Let Keras handle the initial EfficientNetB0 creation with 'imagenet' weights --- 
# This ensures Keras creates the model with correctly initialized Normalization layers from its own trusted source. 
print(f"Loading EfficientNetB0 with 'imagenet' weights (handled by Keras).") 
try: 
    base_model = tf.keras.applications.EfficientNetB0( 
        weights='imagenet', # Let Keras fetch/use its standard imagenet weights (will download if not in cache) 
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3), 
        # REMOVED: name="efficientnetb0_base" # <--- REMOVE THIS LINE 
    ) 
    print("EfficientNetB0 base model created and weights loaded via Keras's built-in mechanism.") 

except Exception as e: 
    # Re-raise as a RuntimeError for clarity, indicating download failed if it's a 403. 
    # This also means if your internet is down, or Google blocks access, this will fail. 
    raise RuntimeError(f"Failed to create EfficientNetB0 model with imagenet weights: {e}. " 
                       "Ensure internet connectivity or that Keras's download is not blocked." 
                       "If blocked, manually place 'efficientnetb0_base_notop.h5' in 'models/' and retry.")
 
 
# 🔒 Step 1: Freeze base model for the first training phase (training custom head) 
base_model.trainable = False 
 
# Build the complete model (without augmentation and normalization layers) 
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image") 
# Data augmentation and normalization are already applied to the dataset in prepare_dataset.py 
x = base_model(inputs, training=False) # Pass through base model (important: training=False when base_model.trainable=False) 
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x) 
x = layers.Dropout(0.3, name="dropout_1")(x) # Add dropout for regularization 
outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x) 
model = models.Model(inputs, outputs, name="dermalscan_model") 
 
model.summary() 
 
# --- Callbacks --- 
# TensorBoard Callback for visualization of training metrics 
log_dir = os.path.join(LOGS_DIR, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 
 
# Model Checkpoint to save the best model based on validation accuracy 
# Saves the entire model in Keras H5 format 
checkpoint_filepath = os.path.join(MODELS_DIR, 'efficientnetb0_skin_model_best.h5') 
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( 
    filepath=checkpoint_filepath, 
    save_weights_only=False, 
    monitor='val_accuracy', 
    mode='max', 
    save_best_only=True, 
    verbose=1 
) 
 
# Early Stopping to prevent overfitting 
early_stopping_callback = tf.keras.callbacks.EarlyStopping( 
    monitor='val_loss', 
    patience=25, # <--- INCREASED PATIENCE 
    restore_best_weights=True, 
    verbose=1 
) 
 
# Reduce Learning Rate on Plateau 
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau( # <--- NEW CALLBACK 
    monitor='val_loss', 
    factor=0.2, # Reduce learning rate by a factor of 0.2 
    patience=5, # Reduce LR after 5 epochs of no improvement 
    min_lr=1e-7, # Don't let learning rate go below this 
    verbose=1 
) 
 
# List of callbacks for the first phase of training 
callbacks_phase1 = [tensorboard_callback, model_checkpoint_callback, early_stopping_callback, reduce_lr_callback] # <--- ADDED reduce_lr_callback 
 
# --- Experimental Class Weights --- 
# Define experimental class weights based on observed poor performance 
# The goal is to encourage the model to predict classes it currently ignores. 
# These weights are subjective and might need tuning. Higher weight = more importance. 
# Class order: 0: 'clear skin', 1: 'dark spots', 2: 'puffy eyes', 3: 'wrinkles' 
# Based on previous evaluation, model was biased towards 'wrinkles' (class 3). 
experimental_class_weights = { 
    0: 0.8,  # clear skin (Increase weight to make model care more about it) 
    1: 0.8,  # dark spots (Increase weight) 
    2: 0.8,  # puffy eyes (Increase weight) 
    3: 2.5   # wrinkles (Lower weight, as model was biased towards it in the last run) 
} 
print("Using experimental class weights:", experimental_class_weights) 
 
 
# --- Phase 1 Training: Train Custom Head --- 
print("\n--- Phase 1: Training custom head (base model frozen) ---") 
model.compile( 
    optimizer=Adam(learning_rate=0.001), 
    loss="categorical_crossentropy", 
    metrics=["accuracy"] 
) 
 
# Train for a few epochs to let the head learn 
initial_epochs = 50 # <--- INCREASED EPOCHS 
history = model.fit( 
    train_ds, 
    epochs=initial_epochs, 
    validation_data=val_ds, 
    callbacks=callbacks_phase1, 
    class_weight=experimental_class_weights # <--- ADDED CLASS WEIGHTS 
) 
 
# Plot training history for Phase 1 
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
    plt.title(f'{phase_name} Training and Validation Accuracy') 
 
    plt.subplot(1, 2, 2) 
    plt.plot(epochs_range, loss, label='Training Loss') 
    plt.plot(epochs_range, val_loss, label='Validation Loss') 
    plt.legend(loc='upper right') 
    plt.title(f'{phase_name} Training and Validation Loss') 
    plt.savefig(os.path.join(HISTORY_PLOT_DIR, f"{phase_name.replace(' ', '_').lower()}{filename_suffix}.png")) 
    plt.show() 
 
plot_history(history.history, "Phase 1: Head Training") 
 
 
# --- Phase 2: Fine-tuning --- 
print("\n--- Phase 2: Fine-tuning (unfreezing top layers of base model) ---") 
# 🔓 Unfreeze a portion of the base model for fine-tuning 
base_model.trainable = True 
 
# Freeze all layers except the last few for fine-tuning 
for layer in base_model.layers[:-40]: # <--- UNFROZE MORE LAYERS (40 instead of 20) 
    layer.trainable = False 
 
model.summary() 
 
# Recompile with a much lower learning rate for fine-tuning 
model.compile( 
    optimizer=Adam(learning_rate=1e-5), 
    loss="categorical_crossentropy", 
    metrics=["accuracy"] 
) 
 
# Continue training for more epochs (fine-tune) 
fine_tune_epochs = 50 # <--- INCREASED EPOCHS 
total_epochs = len(history.epoch) + fine_tune_epochs 
 
history_fine = model.fit( 
    train_ds, 
    epochs=total_epochs, 
    initial_epoch=len(history.epoch), 
    validation_data=val_ds, 
    callbacks=callbacks_phase1, 
    class_weight=experimental_class_weights # <--- ADDED CLASS WEIGHTS 
) 
 
# Combine history for plotting the full training process 
combined_history = {} 
for key in history.history: 
    combined_history[key] = history.history[key] + history_fine.history[key] 
 
plot_history(combined_history, "Full Training (Phase 1 & 2)", "_full_training") 
 
# --- Final Save --- 
final_model_path = os.path.join(MODELS_DIR, "efficientnetb0_skin_model_final_epoch.h5") 
model.save(final_model_path) 
print(f"\n✅ Model training complete. Final model (last epoch) saved to: {final_model_path}") 
print(f"✅ Best model (based on val_accuracy) saved during training to: {checkpoint_filepath}") 
 
# Optional: Evaluate the best model (loaded from checkpoint) on the test set 
print("\nEvaluating best model on the test set...") 
best_model = tf.keras.models.load_model(checkpoint_filepath) 
test_loss, test_accuracy = best_model.evaluate(test_ds) 
print(f"Test Loss: {test_loss:.4f}") 
print(f"Test Accuracy: {test_accuracy:.4f}")