# File: scripts/clean_dataset_mtcnn_300.py

import os
import shutil
from mtcnn import MTCNN
from tqdm import tqdm # For progress bar

# --- Configuration ---
# THIS IS THE MOST CRITICAL PATH: IT MUST POINT TO YOUR RAW IMAGE DATA.
# IT SHOULD CONTAIN SUBFOLDERS LIKE 'clear skin', 'dark spots', etc.
DATA_RAW_BASE_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_raw\ageing_dataset\DATASET" # <--- VERIFY THIS PATH ON YOUR SYSTEM
# Output directory for face-verified images (intermediate step)
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_verified_faces"
MAX_IMAGES_PER_CLASS = 300 # Limit for initial selection per class

# Initialize MTCNN detector
detector = MTCNN()

print(f"Starting initial cleaning and face verification from: {DATA_RAW_BASE_DIR}")
print(f"Outputting face-verified images to: {OUTPUT_DIR}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through each class (subfolder) in the raw dataset
# Ensure DATA_RAW_BASE_DIR actually exists
if not os.path.exists(DATA_RAW_BASE_DIR):
    raise FileNotFoundError(f"Raw data directory not found: {DATA_RAW_BASE_DIR}. Please check the path.")
if not os.path.isdir(DATA_RAW_BASE_DIR):
    raise NotADirectoryError(f"Raw data path is not a directory: {DATA_RAW_BASE_DIR}. Please check the path.")


for class_name in os.listdir(DATA_RAW_BASE_DIR):
    class_input_path = os.path.join(DATA_RAW_BASE_DIR, class_name)
    class_output_path = os.path.join(OUTPUT_DIR, class_name)

    # Skip if it's not a directory (e.g., if there are files in DATA_RAW_BASE_DIR)
    if not os.path.isdir(class_input_path):
        continue

    os.makedirs(class_output_path, exist_ok=True) # Create class output folder

    images_in_class = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_count = 0
    print(f"\nProcessing class: {class_name}")

    if not images_in_class:
        tqdm.write(f"Warning: No images found in {class_input_path} for class {class_name}, skipping.")
        continue

    # Use tqdm for a nice progress bar
    for img_name in tqdm(images_in_class, desc=f"Verifying {class_name}"):
        if current_count >= MAX_IMAGES_PER_CLASS:
            tqdm.write(f"Reached {MAX_IMAGES_PER_CLASS} images for {class_name}, skipping remaining from this class.")
            break

        img_path = os.path.join(class_input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            tqdm.write(f"Warning: Could not read image {img_path}, skipping.")
            continue

        # Convert to RGB for MTCNN (OpenCV reads in BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        if faces:
            # If faces are detected, save the original image to the output path
            output_img_path = os.path.join(class_output_path, img_name)
            shutil.copy2(img_path, output_img_path) # Use shutil.copy2 to preserve metadata
            current_count += 1
        # else:
            # tqdm.write(f"No face detected in {img_name}, skipping.") # Uncomment for more verbose output

    print(f"Finished {class_name}. Verified {current_count} images with faces.")

print("\nInitial cleaning and face verification complete!")