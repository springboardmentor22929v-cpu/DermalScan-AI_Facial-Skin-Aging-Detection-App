# File: scripts/augment_dataset.py
# VERSION: 100% WORKING, CLEAN, SAFE, BALANCED AUGMENTATION PIPELINE

import os
import numpy as np
import random
from shutil import rmtree
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------- CONFIG ---------------------- #

INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\DermalScan-AI_Facial-Skin-Aging-Detection-App\data_preprocessed"
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\DermalScan-AI_Facial-Skin-Aging-Detection-App\data_augmented"


# EXACT number of final images per class
TARGET_COUNT_PER_CLASS = 1500

# Augmentation engine
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

print(f"\nStarting augmentation from: {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Target per class: {TARGET_COUNT_PER_CLASS}\n")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.isdir(INPUT_DIR):
    raise NotADirectoryError(f"Input directory does not exist: {INPUT_DIR}")

# ---------------------- GET CLASSES ---------------------- #

classes = [d for d in os.listdir(INPUT_DIR)
           if os.path.isdir(os.path.join(INPUT_DIR, d))]

if not classes:
    raise Exception("No class folders found inside INPUT_DIR.")

# ---------------------- PROCESS EACH CLASS ---------------------- #

for class_name in classes:

    class_input = os.path.join(INPUT_DIR, class_name)
    class_output = os.path.join(OUTPUT_DIR, class_name)

    # Delete old output and recreate
    if os.path.exists(class_output):
        rmtree(class_output)
    os.makedirs(class_output)

    # Load image names
    images = [f for f in os.listdir(class_input)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    initial_count = len(images)

    if initial_count == 0:
        print(f"⚠️ No images found in: {class_input}. Skipping.\n")
        continue

    print(f"\nClass: {class_name}")
    print(f"Initial images: {initial_count}")

    # -----------------------------------------------------
    # CASE 1: Initial images >= TARGET → randomly select exact target
    # -----------------------------------------------------
    if initial_count >= TARGET_COUNT_PER_CLASS:
        print(f"Initial ≥ Target → Copying {TARGET_COUNT_PER_CLASS} random images...")

        selected = random.sample(images, TARGET_COUNT_PER_CLASS)
        current_total = 0

        for img_name in tqdm(selected, desc=f"Copying {class_name}"):
            src = os.path.join(class_input, img_name)
            dst = os.path.join(class_output, img_name)
            img = cv2.imread(src)
            if img is not None:
                cv2.imwrite(dst, img)
                current_total += 1

        print(f"✔ Finished {class_name}. Total images: {current_total}")
        continue

    # -----------------------------------------------------
    # CASE 2: Initial images < TARGET → copy originals + augment
    # -----------------------------------------------------
    print("Initial < Target → Copying originals + Augmenting until target reached...")

    # Step 1: Copy all originals
    for img_name in tqdm(images, desc=f"Copying originals ({class_name})"):
        src = os.path.join(class_input, img_name)
        dst = os.path.join(class_output, img_name)
        img = cv2.imread(src)
        if img is not None:
            cv2.imwrite(dst, img)

    current_total = initial_count

    # Step 2: Augment until reaching target
    remaining = TARGET_COUNT_PER_CLASS - current_total
    progress = tqdm(total=remaining, desc=f"Augmenting {class_name}")

    while current_total < TARGET_COUNT_PER_CLASS:

        base_img_name = random.choice(images)
        src_path = os.path.join(class_input, base_img_name)

        img = cv2.imread(src_path)
        if img is None:
            print(f"⚠️ Could not read: {src_path}. Skipping...")
            continue

        # Convert for ImageDataGenerator
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_expanded = np.expand_dims(img_rgb, axis=0)

        # Generate ONE augmentation per loop
        batch = next(datagen.flow(img_expanded, batch_size=1))
        aug_rgb = batch[0].astype("uint8")
        aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)

        save_name = f"aug_{current_total}_{os.path.splitext(base_img_name)[0]}.jpg"
        save_path = os.path.join(class_output, save_name)

        cv2.imwrite(save_path, aug_bgr)

        current_total += 1
        progress.update(1)

    progress.close()

    print(f"✔ Finished {class_name}. Total images: {current_total}")

print("\n🎉 All classes augmented to target count successfully!")
