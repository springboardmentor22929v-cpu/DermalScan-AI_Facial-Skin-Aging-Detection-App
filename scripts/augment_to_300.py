# File: scripts/augment_dataset.py (FIXED NAMED AUGMENTATIONS version)

import os
import numpy as np
from tqdm import tqdm
from shutil import rmtree # Import rmtree to delete directories
from PIL import Image

# --- Configuration ---
# Input directory for already preprocessed (cropped, resized, RGB) images
INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_preprocessed"
# Output directory for all augmented images
OUTPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_augmented_fixed" # IMPORTANT: Changed output dir name to distinguish

# Augmentations to apply to EACH image
# Define explicit augmentation functions instead of ImageDataGenerator for precise naming
def augment_horizontal_flip(image_np):
    return cv2.flip(image_np, 1) # 1 means horizontal flip

def augment_rotate(image_np, angle=15):
    (h, w) = image_np.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # Adjust bounding box for new image size (important to prevent cropping parts)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image_np, M, (nW, nH), borderMode=cv2.BORDER_REFLECT) # Use BORDER_REFLECT to fill empty areas

def augment_zoom(image_np, zoom_factor=0.9): # 0.9 means zoom out (scale down), 1.1 would be zoom in (scale up)
    h, w = image_np.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    
    zoomed_image = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Place zoomed image onto original size canvas, filling with reflected pixels
    padded_image = cv2.copyMakeBorder(zoomed_image, start_y, h - new_h - start_y, start_x, w - new_w - start_x, cv2.BORDER_REFLECT)
    return padded_image

print(f"Starting fixed-augmentation from: {INPUT_DIR}")
print(f"Outputting augmented images to: {OUTPUT_DIR} (Each image gets fixed augmentations)")

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

    # Clean and recreate the output folder for this class
    if os.path.exists(class_output_path):
        rmtree(class_output_path)
    os.makedirs(class_output_path)

    images = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) == 0: # Corrected from `initial_count` to `len(images)`
        print(f"Warning: No images found in {class_input_path} for class {class_name}, skipping augmentation.")
        continue

    print(f"\nClass: {class_name} | Initial images: {len(images)}")
    
    processed_count = 0
    for img_name in tqdm(images, desc=f"Augmenting {class_name}"):
        img_path = os.path.join(class_input_path, img_name)
        img_np = cv2.imread(img_path)

        if img_np is None:
            tqdm.write(f"Warning: Could not read image {img_path}, skipping.")
            continue
        
        # Ensure image is 224x224 before augmenting, as rotation/zoom assumes this for consistent output
        img_np_resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        base_name, _ = os.path.splitext(img_name)

        # Save original image (resized to 224x224)
        cv2.imwrite(os.path.join(class_output_path, f"{base_name}_orig.jpg"), img_np_resized)
        processed_count += 1

        # Apply and save Horizontal Flip
        flipped_img = augment_horizontal_flip(img_np_resized)
        cv2.imwrite(os.path.join(class_output_path, f"{base_name}_flip.jpg"), flipped_img)
        processed_count += 1

        # Apply and save Rotation
        rotated_img = augment_rotate(img_np_resized)
        # Resizing the rotated image back to original size (224x224)
        rotated_img_final = cv2.resize(rotated_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(class_output_path, f"{base_name}_rot.jpg"), rotated_img_final)
        processed_count += 1

        # Apply and save Zoom
        zoomed_img = augment_zoom(img_np_resized)
        # Resizing the zoomed image back to original size (224x224)
        zoomed_img_final = cv2.resize(zoomed_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(class_output_path, f"{base_name}_zoom.jpg"), zoomed_img_final)
        processed_count += 1

    print(f"Finished {class_name}. Total images generated: {processed_count} (Original + 3 Augmented per original)")

print("\nFixed-augmentation complete!")