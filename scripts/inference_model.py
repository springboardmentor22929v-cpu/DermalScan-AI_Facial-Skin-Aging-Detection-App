# File: scripts/inference_model.py (MODIFIED for Age Prediction)

import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from PIL import Image, ImageDraw, ImageFont
import os # For drawing functions not available in PIL, if needed

# --- Configuration for DermalScan Model ---
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models"
DERMALSCAN_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_skin_model_best.h5')
IMG_SIZE = 224 # Must match your DermalScan model's input size
DERMALSCAN_CLASS_NAMES = ['clear skin', 'dark spots', 'puffy eyes', 'wrinkles']
MARGIN_RATIO = 0.2 # Margin to expand face bounding box for cropping

# --- Configuration for Age Prediction Model (Placeholder/Simplified) ---
# NOTE: For a real age prediction, you would need a separate trained model (.h5 file)
# For this feasible path, we will create a *dummy* age prediction model for demonstration
# or use a simplified pre-trained model if one is directly available via Keras Applications.
# --- FEASIBILITY ASSUMPTION ---
# We'll simulate a simple age prediction, OR point to a specific pre-trained model.
# Let's assume you've found a simple Keras age prediction model.
# For a fully functional age prediction, you'd integrate a model like:
# 'https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/weights.28-3.73.hdf5'
# For now, we'll create a *dummy* age predictor to show the integration.

# If you want to use a specific pre-trained age model:
# AGE_MODEL_PATH = os.path.join(MODELS_DIR, 'age_gender_model_weights.h5') # You would download this manually
# AGE_IMG_SIZE = 64 # Common input size for some age models

# --- Initialize DermalScan Model ---
try:
    dermalscan_model = tf.keras.models.load_model(DERMALSCAN_MODEL_PATH)
    print(f"DermalScan Model loaded successfully from {DERMALSCAN_MODEL_PATH}")
except Exception as e:
    print(f"Error loading DermalScan Model from {DERMALSCAN_MODEL_PATH}: {e}")
    print("Please ensure the DermalScan model is trained and saved correctly.")
    dermalscan_model = None

try:
    # A tiny placeholder model that just outputs a random number
    # We don't need GlobalAveragePooling2D if we're just creating a truly dummy model
    # Let's make it simpler, or just create it and compile it.
    age_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)), # Specify input shape
        tf.keras.layers.Flatten(), # Flatten the input
        tf.keras.layers.Dense(1, activation='linear') # Simple dense layer for age
    ])
    # The UserWarning above is about InputLayer in Sequential model. Let's fix that.
    # Or even simpler, just print a message since it's a dummy random age.

    # Let's just create a functional API model or a very simple sequential without input layer warnings
    # For a pure dummy, we just need a non-None object.
    class DummyAgeModel:
        def predict(self, input_data, verbose=0):
            import random
            return np.array([[random.randint(20, 70)]]) # Always return a random age

    age_model = DummyAgeModel() # Assign the dummy class instance
    print("Dummy Age Prediction Model initialized (random output).")

except Exception as e:
    print(f"Error initializing dummy age model: {e}")
    age_model = None
# Initialize MTCNN detector
mtcnn_detector = MTCNN()

# --- Helper Function for Preprocessing (matching your training pipeline) ---
def preprocess_image_for_model(face_image_pil, target_size=IMG_SIZE):
    # Resize and normalize
    face_image_pil = face_image_pil.resize((target_size, target_size), Image.LANCZOS)
    face_image_array = np.array(face_image_pil).astype(np.float32)
    face_image_array = face_image_array / 255.0  # Normalize to [0, 1]
    face_image_array = np.expand_dims(face_image_array, axis=0) # Add batch dimension
    return face_image_array

# --- Main Prediction Function ---
def predict_aging_signs(input_image_pil):
    if dermalscan_model is None:
        draw_error = ImageDraw.Draw(input_image_pil)
        draw_error.text((10, 10), "ERROR: DermalScan Model not loaded!", fill=(255,0,0))
        return input_image_pil, [{"error": "DermalScan Model not loaded."}]

    # Ensure input image is RGB
    if input_image_pil.mode != 'RGB':
        input_image_pil = input_image_pil.convert('RGB')

    original_image_np = np.array(input_image_pil)
    draw = ImageDraw.Draw(input_image_pil)
    predictions_list = []

    # Detect faces using MTCNN
    faces = mtcnn_detector.detect_faces(original_image_np)

    if not faces:
        draw_no_face = ImageDraw.Draw(input_image_pil)
        font = ImageFont.truetype("arial.ttf", 20) if os.path.exists("arial.ttf") else ImageFont.load_default()
        draw_no_face.text((10, 10), "No face detected.", fill=(255,165,0), font=font)
        return input_image_pil, [{"message": "No face detected in the image."}]

    # Sort faces by size (largest first)
    faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)

    for i, face in enumerate(faces):
        x, y, width, height = face['box']

        # Expand bounding box slightly for better context
        x1 = max(0, int(x - width * MARGIN_RATIO))
        y1 = max(0, int(y - height * MARGIN_RATIO))
        x2 = min(input_image_pil.width, int(x + width * (1 + MARGIN_RATIO)))
        y2 = min(input_image_pil.height, int(y + height * (1 + MARGIN_RATIO)))

        cropped_face_pil = input_image_pil.crop((x1, y1, x2, y2))

        # --- DermalScan Model Prediction ---
        processed_face_dermalscan = preprocess_image_for_model(cropped_face_pil, target_size=IMG_SIZE)
        dermalscan_prediction = dermalscan_model.predict(processed_face_dermalscan, verbose=0)[0]
        predicted_class_idx = np.argmax(dermalscan_prediction)
        dermalscan_confidence = float(dermalscan_prediction[predicted_class_idx])
        dermalscan_label = DERMALSCAN_CLASS_NAMES[predicted_class_idx]

        # --- Age Prediction Model Prediction ---
        predicted_age = "N/A" # Default if age model is not loaded
        if age_model:
            # For this dummy age model, we'll just use a random age
            # For a real age model, you'd use:
            # processed_face_age = preprocess_image_for_model(cropped_face_pil, target_size=AGE_IMG_SIZE)
            # age_prediction = age_model.predict(processed_face_age, verbose=0)[0][0]
            # predicted_age = int(np.clip(age_prediction, 0, 100)) # Clip age to reasonable range

            predicted_age = random.randint(20, 70) # <--- DUMMY RANDOM AGE FOR DEMO
            predicted_age = f"{predicted_age}"


        predictions_list.append({
            "box": [x1, y1, x2, y2],
            "dermalscan_label": dermalscan_label,
            "dermalscan_confidence": dermalscan_confidence,
            "predicted_age": predicted_age
        })

        # --- Annotate Image ---
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Prepare text for label, confidence, and age
        text_line1 = f"{dermalscan_label}: {dermalscan_confidence*100:.1f}%"
        text_line2 = f"Age: {predicted_age}"
        
        try:
            font_size = max(12, min(25, int(input_image_pil.width / 40)))
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw background for text
        text_bbox1 = draw.textbbox((0,0), text_line1, font=font)
        text_bbox2 = draw.textbbox((0,0), text_line2, font=font)
        
        text_width = max(text_bbox1[2] - text_bbox1[0], text_bbox2[2] - text_bbox2[0])
        text_height = (text_bbox1[3] - text_bbox1[0]) + (text_bbox2[3] - text_bbox2[0]) + 5 # 5 for spacing

        text_x = x1
        text_y = y1 - text_height - 10
        if text_y < 0:
            text_y = y1 + 5

        draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill="red")
        draw.text((text_x + 2, text_y + 2), text_line1, font=font, fill="white")
        draw.text((text_x + 2, text_y + 2 + (text_bbox1[3] - text_bbox1[0]) + 2), text_line2, font=font, fill="white")


    return input_image_pil, predictions_list

# --- Example Usage (for testing the script directly) ---
if __name__ == "__main__":
    print("\n--- Running inference_model.py for direct test (with dummy age) ---")
    # For a real test, replace this with a path to one of your test images
    TEST_IMAGE_PATH = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\data_split_fixed\test\clear skin\clear skin_0_0_augmented_fixed_orig.jpg" # Example path, adjust as needed
    
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"Loading real image from: {TEST_IMAGE_PATH}")
        real_img = Image.open(TEST_IMAGE_PATH)
        annotated_real_img, real_results = predict_aging_signs(real_img)
        if annotated_real_img:
            annotated_real_img.save("test_prediction_output_with_age.jpg")
            print("Test prediction saved to test_prediction_output_with_age.jpg")
            print("Prediction results:", real_results)
    else:
        print(f"Test image not found at {TEST_IMAGE_PATH}. Creating dummy image for test.")
        # Create a dummy image if the test path doesn't exist
        dummy_image = Image.new('RGB', (500, 500), color = 'white')
        draw_dummy = ImageDraw.Draw(dummy_image)
        # Add a simple face-like shape for MTCNN to detect
        draw_dummy.ellipse([100, 100, 400, 400], fill=(200, 150, 150), outline="black", width=2)
        draw_dummy.ellipse([180, 180, 220, 220], fill="black")
        draw_dummy.ellipse([280, 180, 320, 220], fill="black")
        draw_dummy.arc([200, 250, 300, 300], 0, 180, fill="black", width=3)

        annotated_img, results = predict_aging_signs(dummy_image)
        if annotated_img:
            annotated_img.save("test_prediction_output_with_age.jpg")
            print("Dummy image prediction saved to test_prediction_output_with_age.jpg")
            print("Prediction results:", results)
        else:
            print("Prediction failed with dummy image.")