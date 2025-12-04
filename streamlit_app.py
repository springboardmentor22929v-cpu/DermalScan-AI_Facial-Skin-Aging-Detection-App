# File: streamlit_app.py 

import streamlit as st 
import io 
from PIL import Image, ImageDraw, ImageFont 
import numpy as np 
import tensorflow as tf 
from mtcnn import MTCNN 
import os 

# --- Configuration --- 
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\dermalscan-project\models"
DERMALSCAN_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_skin_model_best.h5')
IMG_SIZE = 224
DERMALSCAN_CLASS_NAMES = ['clear skin', 'dark spots', 'puffy eyes', 'wrinkles']
MARGIN_RATIO = 0.2

# --- Streamlit Page Configuration --- 
st.set_page_config(
    page_title="DermalScan AI - Facial Skin Aging Detection",
    page_icon="✨",
    layout="wide"
)

# --- Load Models --- 
@st.cache_resource
def load_dermalscan_model():
    try:
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(DERMALSCAN_MODEL_PATH)
        st.success("DermalScan Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"Error Loading Skin Model: {e}")
        return None

@st.cache_resource
def load_mtcnn_detector():
    try:
        detector = MTCNN()
        st.success("Face Detector Initialized")
        return detector
    except Exception as e:
        st.error(f"Error Initializing MTCNN: {e}")
        return None

# --- Preprocessing --- 
def preprocess_image_for_model(face_image_pil, target_size=IMG_SIZE):
    face_image_pil = face_image_pil.resize((target_size, target_size), Image.LANCZOS)
    face_image_array = np.array(face_image_pil).astype(np.float32) / 255.0
    face_image_array = np.expand_dims(face_image_array, axis=0)
    return face_image_array

# --- Prediction Pipeline --- 
def predict_aging_signs_and_annotate(input_image_pil):
    dermalscan_model = load_dermalscan_model()
    mtcnn_detector = load_mtcnn_detector()

    if dermalscan_model is None or mtcnn_detector is None:
        return input_image_pil, []

    if input_image_pil.mode != 'RGB':
        input_image_pil = input_image_pil.convert('RGB')

    original_image_np = np.array(input_image_pil)
    draw_img_pil = input_image_pil.copy()
    draw = ImageDraw.Draw(draw_img_pil)

    predictions_list = []

    faces = mtcnn_detector.detect_faces(original_image_np)

    if not faces:
        st.warning("No Face Detected")
        return draw_img_pil, [{"message": "No face detected"}]

    faces.sort(key=lambda x: x['box'][2] * x['box'][3], reverse=True)

    for face in faces:
        x, y, width, height = face['box']

        x1 = max(0, int(x - width * MARGIN_RATIO))
        y1 = max(0, int(y - height * MARGIN_RATIO))
        x2 = min(input_image_pil.width, int(x + width * (1 + MARGIN_RATIO)))
        y2 = min(input_image_pil.height, int(y + height * (1 + MARGIN_RATIO)))

        cropped_face = input_image_pil.crop((x1, y1, x2, y2))

        processed_face = preprocess_image_for_model(cropped_face)
        prediction_raw = dermalscan_model.predict(processed_face, verbose=0)[0]

        all_preds = []
        for idx, conf in enumerate(prediction_raw):
            all_preds.append({
                "label": DERMALSCAN_CLASS_NAMES[idx],
                "confidence": float(conf)
            })

        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        predictions_list.append({
            "box": [x1, y1, x2, y2],
            "all_predictions": all_preds
        })

        top_label = all_preds[0]["label"]
        top_conf = all_preds[0]["confidence"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text_line = f"{top_label}: {top_conf*100:.1f}%"

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        draw.text((x1, y1 - 25), text_line, fill="red", font=font)

    return draw_img_pil, predictions_list

# --- Streamlit UI --- 
st.title("✨ DermalScan AI: Facial Skin Aging Detection")
st.markdown("---")
st.write("Upload your image to analyze skin conditions (wrinkles, dark spots, puffy eyes, clear skin).")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    
    # ↓↓↓ MEDIUM SIZE = 300 ↓↓↓
    st.image(image, caption="Uploaded Image", width=300)
    
    st.markdown("---")

    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            annotated_image, predictions = predict_aging_signs_and_annotate(image)

            if predictions and "message" not in predictions[0]:
                st.subheader("Results")

                for pred in predictions:
                    st.write("Detected Skin Conditions:")
                    for p in pred["all_predictions"]:
                        st.write(f"- **{p['label']}**: {p['confidence']*100:.2f}%")

                # ↓↓↓ MEDIUM SIZE OUTPUT = 300 ↓↓↓
                st.image(annotated_image, caption="Prediction Output", width=300)

            else:
                st.warning("No face detected. Try another image.")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("DermalScan AI detects facial skin aging signs using deep learning.")
