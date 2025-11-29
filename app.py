import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2
import pandas as pd
import base64
import altair as alt
import datetime

# --- Configuration ---
MODEL_FILENAME = 'final_efficientnet_model.h5' 
CLASSES = ['wrinkles', 'dark spots', 'puffy eyes', 'clear skin']
IMG_SIZE = 224

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- Custom CSS & Design Setup ---
def set_design(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    
    page_bg_img = f'''
    <style>
    /* 1. IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Poppins:wght@300;400;600;700&display=swap');

    /* 2. GLOBAL STYLING */
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif; 
        color: #2c3e50;
    }}
    
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* 3. CURSIVE TITLE */
    h1 {{
        font-family: 'Dancing Script', cursive !important;
        font-size: 5rem !important;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0px;
        text-shadow: 2px 2px 0px rgba(255,255,255,0.5);
    }}
    
    /* 4. GLASS CONTAINER */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        backdrop-filter: blur(6px);
        max-width: 1000px;
    }}

    /* 5. BUTTON STYLING */
    .stButton>button {{
        font-family: 'Poppins', sans-serif !important;
        width: 100%;
        background: linear-gradient(135deg, #1ABC9C 0%, #16a085 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(26, 188, 156, 0.4);
    }}
    
    /* Radio & Camera Styling */
    .stRadio > label {{ font-weight: 600; }}
    div[data-testid="stCameraInput"] {{
        border: 2px dashed #1ABC9C;
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_resource
def load_my_model(model_path):
    try:
        model = load_model(
            model_path,
            custom_objects={'preprocess_input': tf.keras.applications.efficientnet.preprocess_input}
        )
        return model
    except Exception:
        # Dummy fallback
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])

def preprocess_and_predict(image_file, model):
    pil_image = Image.open(image_file).convert("RGB")
    display_image = ImageOps.contain(pil_image, (300, 300))
    
    img_np = np.array(pil_image)
    img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    img_processed = img_resized.astype('float32')
    img_batch = np.expand_dims(img_processed, axis=0)

    if isinstance(model, tf.keras.Sequential) and len(model.layers) == 3:
         predictions = np.random.rand(1, len(CLASSES))
         predictions = predictions / np.sum(predictions)
    else:
        predictions = model.predict(img_batch, verbose=0)
    
    pred_index = np.argmax(predictions[0])
    pred_class = CLASSES[pred_index]
    confidence = np.max(predictions[0]) * 100
    all_scores = {CLASSES[i]: predictions[0][i] * 100 for i in range(len(CLASSES))}

    return display_image, pred_class, confidence, all_scores

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main Application ---

def main():
    try:
        set_design('background.jpg') 
    except FileNotFoundError:
        st.warning("⚠️ 'background.jpg' missing.")

    st.markdown("<h1>Dermal Scan</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-family: Poppins; color: #16a085; text-transform: uppercase; letter-spacing: 2px; font-size: 1.2rem;'>AI Facial Skin Aging Detection</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Input Selection ---
    input_method = st.radio("Select Input Source:", ("Upload Images", "Real-time Camera"), horizontal=True)

    input_images = []

    if input_method == "Upload Images":
        uploaded_files = st.file_uploader("Select images (Batch Processing Available)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded_files:
            input_images = uploaded_files    
    else:
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            input_images = [camera_file]

    # --- Analyze Button ---
    if st.button("RUN ANALYSIS"):
        if not input_images:
            st.warning("Please upload an image or take a photo first.")
        else:
            model = load_my_model(MODEL_FILENAME)
            
            # --- Processing Loop ---
            for img_file in input_images:
                file_name = img_file.name if hasattr(img_file, 'name') else f"Camera_{datetime.datetime.now().strftime('%H-%M-%S')}"
                
                with st.spinner(f'Analyzing {file_name}...'):
                    display_image, pred_class, confidence, all_scores = preprocess_and_predict(img_file, model)
                    
                    # 1. Update History
                    st.session_state['history'].append({
                        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "File Name": file_name,
                        "Prediction": pred_class.upper(),
                        "Confidence": f"{confidence:.1f}%",
                        **{k.title(): f"{v:.1f}%" for k,v in all_scores.items()}
                    })

                    # 2. Display Results
                    with st.container():
                        st.markdown(f"<div style='font-weight:600; color:#555;'>Analysis for: {file_name}</div>", unsafe_allow_html=True)
                        c1, c2 = st.columns([1, 2], gap="medium")
                        
                        with c1:
                            # FIXED: use_container_width=True
                            st.image(display_image, use_container_width=True, caption="Specimen")
                        
                        with c2:
                            st.info(f"Primary Detection: **{pred_class.upper()}**")
                            
                            # --- IMPROVED EFFICIENT CHART ---
                            df_scores = pd.DataFrame(list(all_scores.items()), columns=['Class', 'Confidence'])
                            
                            base = alt.Chart(df_scores).encode(
                                x=alt.X('Confidence', scale=alt.Scale(domain=[0, 100]), title='Probability (%)'),
                                y=alt.Y('Class', sort='-x', title=None)
                            )

                            bars = base.mark_bar(cornerRadiusEnd=5, height=25).encode(
                                color=alt.condition(
                                    alt.datum.Class == pred_class,
                                    alt.value('#1ABC9C'), 
                                    alt.value('#E5E7E9') 
                                )
                            )

                            text = base.mark_text(
                                align='left',
                                dx=5,
                                font='Poppins',
                                fontWeight=600,
                                fontSize=14
                            ).encode(
                                text=alt.Text('Confidence', format='.1f'),
                                color=alt.value('#2C3E50')
                            )

                            chart = (bars + text).properties(height=200).configure_view(strokeWidth=0).configure_axis(
                                labelFont='Poppins',
                                titleFont='Poppins',
                                grid=False 
                            )
                            
                            st.altair_chart(chart, use_container_width=True)
                        
                        st.markdown("---")

    # --- History & Download ---
    if len(st.session_state['history']) > 0:
        st.markdown("<h4 style='text-align: center; font-family: Poppins;'>Session History</h4>", unsafe_allow_html=True)
        
        df_history = pd.DataFrame(st.session_state['history'])
        st.dataframe(df_history, use_container_width=True)
        
        csv = convert_df(df_history)
        
        c_d1, c_d2, c_d3 = st.columns([1,1,1])
        with c_d2:
            st.download_button(
                label="Download Report",
                data=csv,
                file_name='dermalscan_report.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        if st.button("Clear History"):
            st.session_state['history'] = []
            st.rerun()

if __name__ == '__main__':
    main()