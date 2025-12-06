# File: streamlit_app.py 
 
import streamlit as st 
import io 
from PIL import Image, ImageDraw, ImageFont # ImageFont for drawing text 
import numpy as np 
import tensorflow as tf 
from mtcnn import MTCNN 
import os 
import random # For dummy age model 
 
# --- Configuration --- 
MODELS_DIR = r"C:\Users\dhana\OneDrive\Desktop\DERMALSCAN-AI_FACIAL-SKIN-AGING-DETECTION-APP\models"

# This is the path to your *trained* DermalScan model. 
DERMALSCAN_MODEL_PATH = os.path.join(MODELS_DIR, 'efficientnetb0_skin_model_best.h5') 
IMG_SIZE = 224 # Must match your DermalScan model's input size 
DERMALSCAN_CLASS_NAMES = ['clear skin', 'dark spots', 'puffy eyes', 'wrinkles'] 
MARGIN_RATIO = 0.2 # Margin to expand face bounding box for cropping 
 
# --- Streamlit Page Configuration --- 
st.set_page_config( 
    page_title="DermalScan AI - Facial Skin Aging Detection", 
    page_icon="✨", 
    layout="wide", 
    initial_sidebar_state="expanded" 
) 
 
# --- Custom CSS for a more professional look --- 
st.markdown(""" 
<style> 
    .reportview-container .main .block-container{ 
        padding-top: 2rem; 
        padding-right: 2rem; 
        padding-left: 2rem; 
        padding-bottom: 2rem; 
    } 
    .stButton>button { 
        font-size: 1.2rem; 
        font-weight: bold; 
        color: white; 
        background-color: #4CAF50; /* Green */ 
        border-radius: 0.5rem; 
        border: none; 
        padding: 0.5rem 1rem; 
        margin: 0.5rem 0; 
        cursor: pointer; 
    } 
    .stButton>button:hover { 
        background-color: #45a049; 
    } 
    .stAlert { 
        border-radius: 0.5rem; 
    } 
    h1 { 
        color: #4CAF50; 
        text-align: center; 
        margin-bottom: 1rem; 
    } 
    h2 { 
        color: #333333; 
        margin-top: 1.5rem; 
        margin-bottom: 0.8rem; 
    } 
    h3 { 
        color: #555555; 
        margin-top: 1rem; 
        margin-bottom: 0.5rem; 
    } 
    .uploadedFile { 
        background-color: #e0ffe0; 
        border-left: 5px solid #4CAF50; 
        padding: 10px; 
        border-radius: 5px; 
    } 
    .stExpander { 
        border: 1px solid #ddd; 
        border-radius: 5px; 
        padding: 10px; 
        margin-bottom: 10px; 
    } 
</style> 
""", unsafe_allow_html=True) 
 
 
# --- Model Loading (with Streamlit caching for performance) --- 
@st.cache_resource(show_spinner="Loading DermalScan AI Model...") 
def load_dermalscan_model(): 
    try: 
        # Suppress TensorFlow messages during loading for cleaner UI 
        tf.get_logger().setLevel('ERROR')  
        model = tf.keras.models.load_model(DERMALSCAN_MODEL_PATH) 
        tf.get_logger().setLevel('INFO') # Reset logger level after loading 
        st.success(f"DermalScan AI Model loaded successfully.") 
        return model 
    except Exception as e: 
        st.error(f"Error loading DermalScan AI Model: {e}") 
        st.warning("Please ensure the model is trained and 'efficientnetb0_skin_model_best.h5' is in the 'models/' folder.") 
        return None 
 
@st.cache_resource(show_spinner="Initializing Face Detector...") 
def load_mtcnn_detector(): 
    try: 
        detector = MTCNN() 
        st.success("Face Detector initialized.") 
        return detector 
    except Exception as e: 
        st.error(f"Error initializing MTCNN detector: {e}") 
        return None 
 
# --- Dummy Age Prediction Model --- 
class DummyAgeModel: 
    def predict(self, input_data=None, verbose=0): # input_data can be None for dummy 
        return np.array([[random.randint(20, 70)]]) 
 
# --- Helper Function for Preprocessing --- 
def preprocess_image_for_model(face_image_pil, target_size=IMG_SIZE): 
    face_image_pil = face_image_pil.resize((target_size, target_size), Image.LANCZOS) 
    face_image_array = np.array(face_image_pil).astype(np.float32) 
    face_image_array = face_image_array / 255.0  # Normalize to [0, 1] 
    face_image_array = np.expand_dims(face_image_array, axis=0) # Add batch dimension 
    return face_image_array 
 
# --- Main Prediction Function (integrating all logic, Module 6) --- 
def predict_aging_signs_and_annotate(input_image_pil): 
    dermalscan_model = load_dermalscan_model() 
    mtcnn_detector = load_mtcnn_detector() 
    age_model = DummyAgeModel() # Initialize dummy age model 
 
    if dermalscan_model is None or mtcnn_detector is None: 
        return input_image_pil, [] # Return original image and empty predictions on error 
 
    if input_image_pil.mode != 'RGB': 
        input_image_pil = input_image_pil.convert('RGB') 
 
    original_image_np = np.array(input_image_pil) 
    draw_img_pil = input_image_pil.copy() # Make a copy for drawing 
    draw = ImageDraw.Draw(draw_img_pil) # Draw on the copy 
    predictions_list = [] 
 
    # Detect faces using MTCNN 
    faces = mtcnn_detector.detect_faces(original_image_np) 
 
    if not faces: 
        st.warning("No face detected in the image.") # Streamlit specific warning 
        # Draw a message on the image if no face is detected 
        font_size = max(12, min(25, int(input_image_pil.width / 40))) 
        font = ImageFont.truetype("arial.ttf", font_size) if os.path.exists("arial.ttf") else ImageFont.load_default() 
        draw.text((10, 10), "No face detected.", fill=(255,165,0), font=font) 
        return draw_img_pil, [{"message": "No face detected in the image."}] # Return original with message, empty predictions 
 
 
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
 
        # --- DermalScan Model Prediction (Milestone 2, Module 4) --- 
        processed_face_dermalscan = preprocess_image_for_model(cropped_face_pil, target_size=IMG_SIZE) 
        dermalscan_prediction_raw = dermalscan_model.predict(processed_face_dermalscan, verbose=0)[0] 
         
        # Get all 4 predictions explicitly for display 
        all_dermalscan_preds = [] 
        for idx, conf in enumerate(dermalscan_prediction_raw): 
            all_dermalscan_preds.append({ 
                "label": DERMALSCAN_CLASS_NAMES[idx], 
                "confidence": float(conf) 
            }) 
        # Sort by confidence (descending) 
        all_dermalscan_preds.sort(key=lambda x: x['confidence'], reverse=True) 
 
        # --- Age Prediction Model Prediction (Milestone 2, Module 4) --- 
        age_prediction_result = DummyAgeModel().predict(None) # Input is ignored for dummy model 
        predicted_age = int(age_prediction_result[0][0]) 
        predicted_age_str = f"{predicted_age}" 
 
 
        predictions_list.append({ 
            "box": [x1, y1, x2, y2], 
            "all_predictions": all_dermalscan_preds, # Now contains all 4 
            "predicted_age": predicted_age_str 
        }) 
 
        # --- Annotate Image (Milestone 2, Module 4) --- 
        # For image annotation, use the very top prediction 
        top_label = all_dermalscan_preds[0]['label'] 
        top_confidence = all_dermalscan_preds[0]['confidence'] 
         
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3) 
 
        # Prepare text for top prediction and age (for drawing on image) 
        text_line1 = f"{top_label}: {top_confidence*100:.1f}%" 
        text_line2 = f"Age: {predicted_age_str}" 
         
        try: 
            # Load a standard font from Keras's dependencies or system 
            font_size = max(12, min(25, int(input_image_pil.width / 40))) 
            font = ImageFont.truetype("arial.ttf", font_size) 
        except IOError: 
            font = ImageFont.load_default() # Fallback if font loading fails 
         
        # Adjust text position dynamically to avoid going off-image 
        text_bbox1 = draw.textbbox((0,0), text_line1, font=font) 
        text_bbox2 = draw.textbbox((0,0), text_line2, font=font) 
         
        text_width = max(text_bbox1[2] - text_bbox1[0], text_bbox2[2] - text_bbox2[0]) 
        text_height = (text_bbox1[3] - text_bbox1[0]) + (text_bbox2[3] - text_bbox2[0]) + 5 # 5 for spacing 
 
        text_x = x1 
        text_y = y1 - text_height - 10 
        if text_y < 0: 
            text_y = y1 + 5 # Place inside if it would go off top 
 
        draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill="red") 
        draw.text((text_x + 2, text_y + 2), text_line1, font=font, fill="white") 
        draw.text((text_x + 2, text_y + 2 + (text_bbox1[3] - text_bbox1[0]) + 2), text_line2, font=font, fill="white") 
 
 
    return draw_img_pil, predictions_list 
 
# --- Streamlit UI Layout (Milestone 3, Module 5) --- 
st.title("✨ DermalScan AI: Facial Skin Aging Detection App ✨") 
st.markdown("---") 
st.markdown(""" 
Welcome to DermalScan AI! Upload a facial image to receive an advanced analysis of common aging signs 
(wrinkles, dark spots, puffy eyes, clear skin) and an estimated age. 
""") 
 
# Image Uploader 
st.subheader("📤 Upload Your Facial Image") 
uploaded_file = st.file_uploader("Drag and drop an image here or click to browse", type=["jpg", "jpeg", "png"]) 
 
if uploaded_file is not None: 
    # Display the uploaded image 
    image = Image.open(io.BytesIO(uploaded_file.getvalue())) 
    st.image(image, caption="Uploaded Image for Analysis", width=250)

     
    st.markdown("---") 
     
    # Button to trigger prediction 
    if st.button("🔍 Analyze Face", help="Click to start the AI analysis of the uploaded image", type="primary"): 
        with st.spinner("Analyzing image... This may take a few seconds as the AI processes your facial features."): 
            annotated_image, predictions = predict_aging_signs_and_annotate(image) 
 
            if predictions: 
                if predictions[0].get("message") == "No face detected in the image.": 
                    st.warning("No face detected in the image. Please try another image with a clear face.") 
                    st.image(image, caption="Original Image (No Face Detected)", use_column_width=True) # Display original 
                else: 
                    st.subheader("📊 AI Analysis Results:") 
                    for i, pred_data in enumerate(predictions): 
                        st.markdown(f"**Face {i+1} detected (Bounding Box: {pred_data['box'][0]}-{pred_data['box'][1]} to {pred_data['box'][2]}-{pred_data['box'][3]}):**") 
                        st.write("Aging Signs (Confidence for each class):") 
                        for p in pred_data["all_predictions"]: 
                            st.write(f"- **{p['label']}**: {p['confidence']*100:.1f}%") 
                        st.write(f"Age: **{pred_data['predicted_age']}**") 
                        st.markdown("---") # Separator between faces 
                        st.image(annotated_image, caption="Analyzed Image", width=250)

                    
            else: 
                st.error("An unexpected error occurred during prediction or model loading failed. Please check the terminal logs.") 
 
# --- Professional Sidebar --- 
st.sidebar.header("About DermalScan AI") 
st.sidebar.markdown(""" 
DermalScan AI is a prototype web application for facial skin aging detection. 
It leverages advanced deep learning to analyze skin conditions and estimate age. 
""") 
st.sidebar.subheader("How it Works:") 
st.sidebar.markdown(""" 
1. **Upload**: You upload a facial image. 
2. **Detect**: AI (MTCNN) detects faces. 
3. **Analyze**: Our DermalScan model (EfficientNetB0) classifies aging signs. 
4. **Predict**: A dummy age prediction is made. 
5. **Visualize**: Results are shown on an annotated image. 
""") 
st.sidebar.subheader("Disclaimer:") 
st.sidebar.info("This is a prototype. Predictions are for demonstration purposes only and should not be used for medical diagnosis or advice.") 
st.sidebar.markdown("---") 
st.sidebar.caption("Built by Dhanasri Batchu | Powered by TensorFlow & Streamlit")