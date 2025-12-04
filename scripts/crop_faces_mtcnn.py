# File: scripts/crop_faces_mtcnn.py 
 
import os 
import numpy as np 
from mtcnn import MTCNN 
from PIL import Image 
from tqdm import tqdm 
 
# --- Configuration --- 
# Input directory for COLOR CORRECTED images (output of apply_color_correction.py) 
INPUT_DIR = r'C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_corrected_faces' 
# Output directory for cropped and resized images 
OUTPUT_DIR = r'C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_preprocessed' 
IMG_SIZE = 224 # Target size for cropped faces 
 
# Create output folder if it doesn't exist 
os.makedirs(OUTPUT_DIR, exist_ok=True) 
 
# Initialize MTCNN detector 
detector = MTCNN() 
 
print(f"Starting face cropping and resizing from: {INPUT_DIR}") 
print(f"Outputting cropped faces to: {OUTPUT_DIR}") 
 
# Ensure INPUT_DIR actually exists 
if not os.path.exists(INPUT_DIR): 
    raise FileNotFoundError(f"Input directory for cropping not found: {INPUT_DIR}. Run apply_color_correction.py first.") 
if not os.path.isdir(INPUT_DIR): 
    raise NotADirectoryError(f"Input path is not a directory: {INPUT_DIR}. Please check the path.") 
 
# Dynamically get class names from the input directory 
classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))] 
 
if not classes: 
    print(f"No class subdirectories found in {INPUT_DIR}. Skipping cropping.") 
 
for class_name in classes: 
    class_input_path = os.path.join(INPUT_DIR, class_name) 
    class_output_path = os.path.join(OUTPUT_DIR, class_name) 
    os.makedirs(class_output_path, exist_ok=True) 
 
    print(f"\nProcessing class: {class_name}") 
 
    images_in_class = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] 
 
    if not images_in_class: 
        tqdm.write(f"Warning: No images found in {class_input_path} for class {class_name}, skipping.") 
        continue 
 
    for img_name in tqdm(images_in_class, desc=f"Cropping {class_name}"): 
        img_path = os.path.join(class_input_path, img_name) 
        try: 
            # Use PIL for robust image loading and ensure RGB 
            img = Image.open(img_path).convert('RGB') 
        except Exception as e: 
            tqdm.write(f"Warning: Could not open or convert {img_path}: {e}, skipping.") 
            continue 
 
        pixels = np.array(img) # Convert PIL image to numpy array for MTCNN 
        results = detector.detect_faces(pixels) 
 
        if results: 
            # Assume only one main face or take the largest one 
            # Sort by bounding box area (width * height) in descending order 
            results.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True) 
            main_face = results[0] # Take the largest face 
 
            x, y, width, height = main_face['box'] 
 
            # Expand bounding box slightly for better context (optional) 
            margin_ratio = 0.2 # 20% margin around the face 
            x1 = max(0, int(x - width * MARGIN_RATIO)) 
            y1 = max(0, int(y - height * MARGIN_RATIO)) 
            x2 = min(img.width, int(x + width * (1 + margin_ratio))) 
            y2 = min(img.height, int(y + height * (1 + margin_ratio))) 
 
 
            cropped_face = img.crop((x1, y1, x2, y2)) 
            # Resize the cropped face to the target IMG_SIZE 
            cropped_face = cropped_face.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS) # Use LANCZOS for high quality downsampling 
 
            # Save the cropped face as JPG to standardize format 
            base_name, _ = os.path.splitext(img_name) 
            save_path = os.path.join(class_output_path, f"{base_name}_cropped.jpg") 
            cropped_face.save(save_path, "JPEG", quality=90) # Specify JPEG format with quality 
 
        # else: 
            # tqdm.write(f"No face detected in {img_name} for cropping, skipping.") # Uncomment for more verbose output 
 
print("\nFace cropping and resizing completed!")