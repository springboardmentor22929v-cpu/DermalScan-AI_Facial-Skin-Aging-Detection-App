# File: scripts/augment_dataset.py (VERSION TO ENFORCE EXACTLY 300 IMAGES PER CLASS)

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import random
from shutil import rmtree # Import rmtree to delete directories

# --- Configuration ---
# Input directory for already preprocessed (cropped, resized, RGB) images
INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_preprocessed"
# Output directory for all augmented images (including copies of originals)
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_augmented" # Changed output dir name back for consistency if needed

# Number of images per class you want *EXACTLY* after augmentation
TARGET_COUNT_PER_CLASS = 300 # Enforcing exactly 300 images per class

# Augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotate images by up to 20 degrees
    width_shift_range=0.1,    # Shift width by up to 10%
    height_shift_range=0.1,   # Shift height by up to 10%
    zoom_range=0.1,           # Zoom in/out by up to 10%
    horizontal_flip=True,     # Randomly flip images horizontally
    brightness_range=[0.8,1.2], # Randomly adjust brightness
    fill_mode='nearest'       # Strategy for filling in new pixels created by transforms
)

print(f"Starting data augmentation from: {INPUT_DIR}")
print(f"Outputting augmented images to: {OUTPUT_DIR} (Target: {TARGET_COUNT_PER_CLASS} per class)")

# Ensure INPUT_DIR actually exists
if not os.path.exists(INPUT_DIR):
    print(f"Warning: Data preprocessed directory not found: {INPUT_DIR}. Skipping augmentation.")
    exit()
if not os.path.isdir(INPUT_DIR):
    print(f"Warning: Data preprocessed path is not a directory: {INPUT_DIR}. Skipping augmentation.")
    exit()

# Loop through each class
classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

if not classes:
    print(f"No class subdirectories found in {INPUT_DIR}. Skipping augmentation.")

for class_name in classes:
    class_input_path = os.path.join(INPUT_DIR, class_name)
    class_output_path = os.path.join(OUTPUT_DIR, class_name)

    # --- Delete and recreate class_output_path ---
    # This ensures a clean slate before populating
    if os.path.exists(class_output_path):
        rmtree(class_output_path) # Delete existing folder
    os.makedirs(class_output_path) # Recreate empty folder
    # --- END NEW LOGIC ---

    images = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    initial_count = len(images)

    if initial_count == 0:
        print(f"Warning: No images found in {class_input_path} for class {class_name}, skipping augmentation.")
        continue

    print(f"\nClass: {class_name} | Initial images from preprocessed: {initial_count}")

    # --- Handle cases where initial_count > TARGET_COUNT_PER_CLASS ---
    # If initial images are already more than target, we need to randomly select TARGET_COUNT_PER_CLASS
    if initial_count >= TARGET_COUNT_PER_CLASS:
        print(f"  Initial images ({initial_count}) >= Target ({TARGET_COUNT_PER_CLASS}). Selecting {TARGET_COUNT_PER_CLASS} random images.")
        selected_images_for_copy = random.sample(images, TARGET_COUNT_PER_CLASS)
        current_total_images = 0 # Reset count for copying
        for img_name in tqdm(selected_images_for_copy, desc=f"Copying/Selecting for {class_name}"):
            src_path = os.path.join(class_input_path, img_name)
            dst_path = os.path.join(class_output_path, img_name)
            cv2.imwrite(dst_path, cv2.imread(src_path))
            current_total_images += 1
    else:
        # If initial images are less than target, copy all and then augment
        print(f"  Initial images ({initial_count}) < Target ({TARGET_COUNT_PER_CLASS}). Copying all and augmenting.")
        for img_name in tqdm(images, desc=f"Copying originals for {class_name}"):
            src_path = os.path.join(class_input_path, img_name)
            dst_path = os.path.join(class_output_path, img_name)
            cv2.imwrite(dst_path, cv2.imread(src_path))
        current_total_images = initial_count
        
        # Augment until target is reached
        tqdm_bar = tqdm(total=TARGET_COUNT_PER_CLASS - current_total_images, desc=f"Augmenting {class_name}")
        while current_total_images < TARGET_COUNT_PER_CLASS:
            img_name_to_augment = random.choice(images) # Pick a random image from the original set to augment
            img_path_to_augment = os.path.join(class_input_path, img_name_to_augment)
            img = cv2.imread(img_path_to_augment)

            if img is None:
                tqdm.write(f"Warning: Could not read image {img_path_to_augment} during augmentation, skipping.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_expanded = np.expand_dims(img_rgb, 0)

            for batch in datagen.flow(img_expanded, batch_size=1):
                aug_img_rgb = batch[0].astype(np.uint8)
                aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
                save_name = f"aug_{current_total_images}_{os.path.splitext(img_name_to_augment)[0]}.jpg"
                save_path = os.path.join(class_output_path, save_name)

                cv2.imwrite(save_path, aug_img_bgr)
                current_total_images += 1
                tqdm_bar.update(1)

                if current_total_images >= TARGET_COUNT_PER_CLASS:
                    break
        tqdm_bar.close()

    print(f"Finished {class_name}. Total images: {current_total_images}") # This will now always be TARGET_COUNT_PER_CLASS

print("\nAll classes augmented to target count!")