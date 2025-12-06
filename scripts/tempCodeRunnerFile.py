# File: scripts/augment_dataset.py

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import random

# --- Configuration ---
# Input directory for already preprocessed (cropped, resized, RGB) images
INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_preprocessed"
# Output directory for all augmented images (including copies of originals)
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_augmented"

# Number of images per class you want *AFTER* augmentation (including originals)
# Aim for a much larger number, e.g., 5-10x your initial 300 images
TARGET_COUNT_PER_CLASS = 1500 # Example: if you have 300 originals, this creates 1200 new augmented images

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

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    os.makedirs(class_output_path, exist_ok=True)

    images = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    initial_count = len(images)

    if initial_count == 0:
        print(f"Warning: No images found in {class_input_path} for class {class_name}, skipping augmentation.")
        continue

    print(f"\nClass: {class_name} | Initial images: {initial_count}")

    # Copy original images to the augmented directory first
    for img_name in tqdm(images, desc=f"Copying originals for {class_name}"):
        src_path = os.path.join(class_input_path, img_name)
        dst_path = os.path.join(class_output_path, img_name)
        # Use cv2.imread and imwrite to handle consistent image loading/saving
        cv2.imwrite(dst_path, cv2.imread(src_path)) 

    current_total_images = initial_count
    
    # Only augment if target is greater than initial count
    if TARGET_COUNT_PER_CLASS > initial_count:
        tqdm_bar = tqdm(total=TARGET_COUNT_PER_CLASS - initial_count, desc=f"Augmenting {class_name}")

        while current_total_images < TARGET_COUNT_PER_CLASS:
            # Pick a random image from the original set to augment
            img_name_to_augment = random.choice(images)
            img_path_to_augment = os.path.join(class_input_path, img_name_to_augment)
            img = cv2.imread(img_path_to_augment)

            if img is None:
                tqdm.write(f"Warning: Could not read image {img_path_to_augment} during augmentation, skipping.")
                # This could cause an infinite loop if `images` list only has unreadable images
                # Consider removing unreadable images from `images` list
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for ImageDataGenerator
            img_expanded = np.expand_dims(img_rgb, 0) # Add batch dimension

            # Generate one augmented image
            for batch in datagen.flow(img_expanded, batch_size=1):
                aug_img_rgb = batch[0].astype(np.uint8)
                aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV imwrite

                # Generate a unique filename for the augmented image
                save_name = f"aug_{current_total_images}_{os.path.splitext(img_name_to_augment)[0]}.jpg"
                save_path = os.path.join(class_output_path, save_name)

                cv2.imwrite(save_path, aug_img_bgr)
                current_total_images += 1
                tqdm_bar.update(1)

                if current_total_images >= TARGET_COUNT_PER_CLASS:
                    break # Exit inner loop if target reached

        tqdm_bar.close()
    else:
        print(f"Target count {TARGET_COUNT_PER_CLASS} not greater than initial count {initial_count}. No augmentation performed for {class_name}.")


    print(f"Finished {class_name}. Total images (original + augmented): {current_total_images}")

print("\nAll classes augmented to target count!")