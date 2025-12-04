# File: scripts/prepare_dataset.py (MODIFIED to move augmentation to dataset pipeline)

import tensorflow as tf
import os

# --- Configuration ---
# Path to your split dataset (output of split_dataset.py)
DATA_SPLIT_DIR = r'C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_split_fixed'

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
    label_mode='categorical',
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True, # Shuffle training data for better generalization
    interpolation='nearest'
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_SPLIT_DIR, 'val'),
    labels='inferred',
    label_mode='categorical',
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
    label_mode='categorical',
    seed=RANDOM_SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False, # No shuffling for test
    interpolation='nearest'
)

# Get class names from the dataset (they will be in alphabetical order)
class_names = train_ds.class_names
num_classes = len(class_names)

# --- Define Data Augmentation Pipeline for tf.data ---
# These layers will be applied *before* the images go into the model
data_augmentation_pipeline = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
  tf.keras.layers.RandomContrast(factor=0.2),
  tf.keras.layers.RandomBrightness(factor=0.2)
], name="data_augmentation_pipeline")

# --- Apply Normalization and Data Augmentation to the datasets ---
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply augmentation ONLY to the training dataset
# Apply normalization to all datasets
def process_train_sample(image, label):
    image = data_augmentation_pipeline(image, training=True) # Apply augmentation
    image = normalization_layer(image) # Apply normalization
    return image, label

def process_val_test_sample(image, label):
    image = normalization_layer(image) # Only normalization for val/test
    return image, label

train_ds = train_ds.map(process_train_sample, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(process_val_test_sample, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(process_val_test_sample, num_parallel_calls=tf.data.AUTOTUNE)

# --- Autotune for performance ---
AUTOTUNE = tf.data.AUTOTUNE
# Removed .cache() from all datasets
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Datasets loaded successfully!")
print(f"Class names: {class_names}")