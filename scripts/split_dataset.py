# File: scripts/split_dataset.py 
 
import os 
import shutil 
import random 
from sklearn.model_selection import train_test_split 
from tqdm import tqdm 
 
# --- Configuration --- 
# Input directory for all augmented images (output of augment_dataset.py) 
INPUT_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_augmented" 
# Base output directory for the split datasets 
OUTPUT_BASE_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_split" 
 
# Split ratios 
TRAIN_RATIO = 0.70 
VAL_RATIO = 0.15 
TEST_RATIO = 0.15 
RANDOM_SEED = 42 # For reproducibility of splits 
 
# Ensure ratios sum to 1 (approximately) 
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Split ratios must sum to 1.0" 
 
print(f"Splitting dataset from: {INPUT_DIR}") 
print(f"Outputting splits to: {OUTPUT_BASE_DIR}") 
print(f"Train:{TRAIN_RATIO*100}%, Validation:{VAL_RATIO*100}%, Test:{TEST_RATIO*100}%") 
 
# Create output directories for train, val, test 
train_dir = os.path.join(OUTPUT_BASE_DIR, 'train') 
val_dir = os.path.join(OUTPUT_BASE_DIR, 'val') 
test_dir = os.path.join(OUTPUT_BASE_DIR, 'test') 
 
# --- NEW LOGIC: Clean output directories before populating --- 
for d in [train_dir, val_dir, test_dir]: 
    if os.path.exists(d): 
        print(f"Deleting existing directory: {d}") 
        shutil.rmtree(d) 
    os.makedirs(d) # Create fresh empty directory 
# --- END NEW LOGIC --- 
 
# Ensure INPUT_DIR actually exists 
if not os.path.exists(INPUT_DIR): 
    raise FileNotFoundError(f"Input directory for splitting not found: {INPUT_DIR}. Please run augment_dataset.py first.") 
if not os.path.isdir(INPUT_DIR): 
    raise NotADirectoryError(f"Input path is not a directory: {INPUT_DIR}. Please check the path.") 
 
# Loop through each class 
classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))] 
 
if not classes: 
    raise ValueError(f"No class subdirectories found in {INPUT_DIR}. Please check if augment_dataset.py ran correctly.") 
 
for class_name in classes: 
    class_input_path = os.path.join(INPUT_DIR, class_name) 
 
    # Create class subdirectories in output splits 
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True) 
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True) 
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True) 
 
    images_in_class = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 
 
    if not images_in_class: 
        print(f"Warning: No images found for class {class_name} in {class_input_path}, skipping split for this class.") 
        continue 
 
    print(f"\nSplitting class: {class_name} ({len(images_in_class)} images)") 
 
    # First split: (train+val) vs test 
    # test_size calculated as TEST_RATIO relative to total images 
    train_val_images, test_images = train_test_split( 
        images_in_class, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED, 
        shuffle=True 
    ) 
 
    # Second split: train vs val from the train_val_images pool 
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO) 
    train_images, val_images = train_test_split( 
        train_val_images, 
        test_size=val_ratio_adjusted, 
        random_state=RANDOM_SEED, 
        shuffle=True 
    ) 
 
    print(f"  Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}") 
 
    # Define destinations and corresponding image lists 
    datasets = { 
        train_dir: train_images, 
        val_dir: val_images, 
        test_dir: test_images 
    } 
 
    # Copy files to their respective destinations 
    for dest_base_dir, img_list in datasets.items(): 
        dest_class_dir = os.path.join(dest_base_dir, class_name) 
        for img_name in tqdm(img_list, desc=f"Copying to {os.path.basename(dest_base_dir)}/{class_name}"): 
            src_path = os.path.join(class_input_path, img_name) 
            dst_path = os.path.join(dest_class_dir, img_name) 
            shutil.copy2(src_path, dst_path) # Use shutil.copy2 to preserve metadata 
 
print("\nDataset splitting complete!")