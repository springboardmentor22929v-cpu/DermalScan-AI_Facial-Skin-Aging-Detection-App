# File: scripts/prepare_dataset.py

import tensorflow as tf
import os

# --- Configuration ---
# Path to your split dataset (output of split_dataset.py)
DATA_SPLIT_DIR = r'C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_split' # <--- CHANGED THIS LINE

# Image settings (must match what was used for cropping and model input)
IMG_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42 # For reproducibility, should match split_dataset.py

# --- Dataset Loading ---
print(f"Loading datasets from: {DATA_SPLIT_DIR}")

# Ensure DATA_SPLIT_DIR actually exists and contains subfolders
if not os.path.exists(DATA_SPLIT_DIR) or not os.path.isdir(DATA_SPLIT_DIR):
    raise FileNotFoundError(f"Data split directory not found: {DATA_SPLIT_DIR}. Run split_dataset.py first.")

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_SPLIT_DIR, 'train'),
    labels='inferred',
    label_mode='categorical', # Changed to CATEGORICAL for one-hot encoding
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle training data for better generalization
    interpolation='nearest' # Use 'nearest' for resizing during load
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_SPLIT_DIR, 'val'),
    labels='inferred',
    label_mode='categorical', # Changed to CATEGORICAL
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False, # No shuffling for validation
    interpolation='nearest'
)

# Load test dataset (for final evaluation)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_SPLIT_DIR, 'test'),
    labels='inferred',
    label_mode='categorical', # Changed to CATEGORICAL
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False, # No shuffling for test
    interpolation='nearest'
)

# Get class names from the dataset (they will be in alphabetical order)
class_names = train_ds.class_names
num_classes = len(class_names)

# --- Apply Normalization to the datasets here ---
# This means the model will receive normalized data directly
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# --- Autotune for performance ---
AUTOTUNE = tf.data.AUTOTUNE
# Removed .cache() from all datasets to avoid pickling errors
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Datasets loaded successfully!")
print(f"Class names: {class_names}")