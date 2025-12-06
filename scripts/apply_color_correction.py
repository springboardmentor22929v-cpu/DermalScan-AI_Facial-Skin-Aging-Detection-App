# File: scripts/apply_color_correction.py

import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# Input directory for face-verified images (output of clean_dataset_mtcnn_300.py)
INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_verified_faces"
# Output directory for images after color and illumination correction
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_corrected_faces"

# --- Correction Parameters ---
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0  # Threshold for contrast limiting
CLAHE_TILE_GRID_SIZE = (8, 8) # Size of grid for histogram equalization

# Gamma Correction
GAMMA_VALUE_INITIAL = 0.8 # Initial gamma value for brightness normalization
GAMMA_VALUE_FEATURE_ENHANCEMENT = 1.2 # Gamma value for enhancing features

print(f"Starting color and illumination correction from: {INPUT_DIR}")
print(f"Outputting corrected images to: {OUTPUT_DIR}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure INPUT_DIR actually exists
if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input directory for color correction not found: {INPUT_DIR}. Run clean_dataset_mtcnn_300.py first.")
if not os.path.isdir(INPUT_DIR):
    raise NotADirectoryError(f"Input path is not a directory: {INPUT_DIR}. Please check the path.")

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

# Function to apply gamma correction
def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Loop through each class
classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

if not classes:
    print(f"No class subdirectories found in {INPUT_DIR}. Skipping color correction.")

for class_name in classes:
    class_input_path = os.path.join(INPUT_DIR, class_name)
    class_output_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\nProcessing class: {class_name}")

    images_in_class = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images_in_class:
        tqdm.write(f"Warning: No images found in {class_input_path} for class {class_name}, skipping.")
        continue

    for img_name in tqdm(images_in_class, desc=f"Correcting {class_name}"):
        img_path = os.path.join(class_input_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            tqdm.write(f"Warning: Could not read image {img_path}, skipping.")
            continue

        # Convert to LAB color space for CLAHE (it works best on L channel)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Apply Gamma Correction for brightness normalization
        gamma_corrected_img = apply_gamma_correction(clahe_img, GAMMA_VALUE_INITIAL)

        # Apply Gamma Correction for feature enhancement (optional second pass)
        final_corrected_img = apply_gamma_correction(gamma_corrected_img, GAMMA_VALUE_FEATURE_ENHANCEMENT)


        # Save the corrected image
        base_name, _ = os.path.splitext(img_name)
        save_path = os.path.join(class_output_path, f"{base_name}_corrected.jpg")
        cv2.imwrite(save_path, final_corrected_img)

print("\nColor and illumination correction complete!")