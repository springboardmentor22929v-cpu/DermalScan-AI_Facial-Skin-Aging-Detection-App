# File: scripts/convert_to_rgb.py

import os
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Path to your preprocessed dataset (output of crop_faces_mtcnn.py)
DATASET_PATH = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_preprocessed"

print(f"Ensuring all images in {DATASET_PATH} are RGB and standardized to JPEG.")

# Ensure DATASET_PATH actually exists
if not os.path.exists(DATASET_PATH):
    print(f"Warning: Data preprocessed directory not found: {DATASET_PATH}. Skipping RGB conversion.")
    exit()
if not os.path.isdir(DATASET_PATH):
    print(f"Warning: Data preprocessed path is not a directory: {DATASET_PATH}. Skipping RGB conversion.")
    exit()

# Loop through each class folder
classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

if not classes:
    print(f"No class subdirectories found in {DATASET_PATH}. Skipping RGB conversion.")

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)

    print(f"\nProcessing class: {class_name}")

    images_in_class = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images_in_class:
        tqdm.write(f"Warning: No images found in {class_path} for class {class_name}, skipping.")
        continue

    for img_name in tqdm(images_in_class, desc=f"Converting {class_name}"):
        img_path = os.path.join(class_path, img_name)
        try:
            img = Image.open(img_path)
            
            # Check if conversion or format change is needed
            needs_save = False
            if img.mode != 'RGB':
                img = img.convert("RGB")
                needs_save = True
            
            base_name, ext = os.path.splitext(img_name)
            new_img_path = os.path.join(class_path, f"{base_name}.jpg")

            # If it's a PNG or a different format, save as JPG
            if ext.lower() != '.jpg' or needs_save:
                img.save(new_img_path, "JPEG", quality=90) # Save with good quality
                # If original was not JPG, remove the old file (unless it's the same path)
                if img_path != new_img_path:
                    os.remove(img_path)
            
        except Exception as e:
            tqdm.write(f"Error processing {img_path}: {e}")

print("All images in data_preprocessed ensured to be RGB (JPEG) successfully!")