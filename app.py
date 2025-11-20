import io
import os
import sys
import tempfile
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
CLASSES = ['wrinkles', 'dark spots', 'puffy eyes', 'clear skin']
IMG_SIZE = 224
MODEL_PATH = 'final.h5'


logging.basicConfig(level=logging.INFO)


@st.cache_resource
def load_model_cached(path):
    try:
        logging.info(f"Loading model from: {path}")
        m = load_model(path)
        logging.info("Model loaded successfully")
        return m
    except Exception as e:
        logging.exception("Error loading model")
        return None


def preds_to_percent(preds: np.ndarray) -> np.ndarray:
    # Convert model output to percentage per-class in a robust way
    preds = np.array(preds, dtype=float)
    if preds.size == 0:
        return preds
    if preds.max() <= 1.01 and preds.min() >= 0.0:
        # Probably probabilities in [0,1]
        return preds * 100.0
    if preds.max() > 1.0 and preds.max() <= 100.0:
        # Already in percentages
        return preds
    # Fallback: softmax then convert to percent
    e = np.exp(preds - np.max(preds))
    soft = e / e.sum()
###


def run_inference_on_crop(model, crop_rgb):
    # Resize using PIL instead of cv2
    img_pil = Image.fromarray(crop_rgb) if isinstance(crop_rgb, np.ndarray) else crop_rgb
    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img_pil).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    if isinstance(preds, list):
        preds = np.array(preds).squeeze()
    preds = np.squeeze(preds)
    return preds_to_percent(preds)


def main():
    st.set_page_config(page_title="DermalScan", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        * {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-title {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 3em;
            font-weight: bold;
            padding: 1.5em;
            border-radius: 15px;
            margin-bottom: 0.5em;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.2em;
            margin-bottom: 2em;
            font-weight: 500;
        }
        .result-box {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            padding: 1.5em;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            margin: 0.8em 0;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
            transition: transform 0.2s;
        }
        .result-box:hover {
            transform: translateX(5px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.15);
        }
        .class-label {
            font-weight: bold;
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 0.5em;
        }
        .confidence {
            font-size: 1.15em;
            color: #2ca02c;
            font-weight: bold;
        }
        .stat-card {
            background: white;
            padding: 1.2em;
            border-radius: 10px;
            border-top: 4px solid #667eea;
            margin: 0.5em 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .high-confidence {
            color: #2ca02c;
            font-weight: bold;
        }
        .medium-confidence {
            color: #ff7f0e;
            font-weight: bold;
        }
        .low-confidence {
            color: #d62728;
            font-weight: bold;
        }
        .divider {
            border-top: 3px solid #667eea;
            margin: 2em 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🔬 DermalScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Powered Facial Skin Aging Detection</div>', unsafe_allow_html=True)

    # Sidebar: model controls and status
    st.sidebar.header("⚙️ Settings")
    st.sidebar.info("Upload a facial image to analyze skin conditions and receive AI predictions.")

    # Ensure session state holds model info
    if 'model' not in st.session_state:
        st.session_state['model'] = None
        st.session_state['model_path'] = None

    # If no model in session, attempt to load default path silently
    if st.session_state['model'] is None:
        default = MODEL_PATH
        m = load_model_cached(default)
        if m is not None:
            st.session_state['model'] = m
            st.session_state['model_path'] = default

    # Show model status
    if st.session_state['model'] is None:
        st.sidebar.error("❌ Model not loaded. Please ensure `final.h5` is in the app directory.")
    else:
        st.sidebar.success(f"✅ Model ready: {os.path.basename(st.session_state['model_path'])}")

    model = st.session_state['model']

    # Annotations have been removed; always show the original image (no boxes)
    show_annotations = False

    st.markdown("---")
    uploaded = st.file_uploader("📸 Upload image (jpg, png)", type=['jpg', 'jpeg', 'png'])

    if uploaded is not None and model is not None:
        image_bytes = uploaded.read()
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig = np.array(img_pil)

        annotations = []

        with st.spinner('🔍 Analyzing image...'):
            # Run inference on full image (no face detection)
            probs = run_inference_on_crop(model, orig)
            annotations.append({
                'box': (0, 0, orig.shape[1], orig.shape[0]),
                'label': CLASSES[int(np.argmax(probs))],
                'prob': float(np.max(probs)),
                'probs': probs,
            })

        # Annotations removed: always show original image
        image_to_show = Image.fromarray(orig)
        caption = 'Original image (no annotations)'

        # Layout: image left, per-face probabilities on right
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.subheader("📷 Analysis Result")
            # Convert PIL image to bytes and display via bytes buffer.
            buf = io.BytesIO()
            image_to_show.save(buf, format='JPEG')
            buf.seek(0)
            st.image(buf.getvalue(), caption='Uploaded image', use_container_width=True)

        with col2:
            st.subheader("📊 Predictions")
            for i, ann in enumerate(annotations):
                with st.container():
                    st.markdown(f'<div class="result-box">', unsafe_allow_html=True)
                    st.markdown(f'<div class="class-label">🎯 {ann["label"].title()}</div>', unsafe_allow_html=True)
                    
                    # Color code confidence
                    conf = ann['prob']
                    if conf >= 75:
                        conf_class = "high-confidence"
                        conf_emoji = "✅"
                    elif conf >= 50:
                        conf_class = "medium-confidence"
                        conf_emoji = "⚠️"
                    else:
                        conf_class = "low-confidence"
                        conf_emoji = "❌"
                    
                    st.markdown(f'<div class="{conf_class}">Confidence: {conf_emoji} {conf:.1f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed breakdown
                    with st.expander("📈 Detailed Analysis"):
                        probs = ann['probs']
                        
                        # Create a bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors = ['#2ca02c' if j == int(np.argmax(probs)) else '#d3d3d3' for j in range(len(CLASSES))]
                        bars = ax.barh([c.title() for c in CLASSES], probs, color=colors)
                        ax.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
                        ax.set_title(f'Region {i+1}: Confidence Distribution', fontsize=12, fontweight='bold')
                        ax.set_xlim([0, 100])
                        
                        # Add value labels on bars
                        for bar, prob in zip(bars, probs):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                   f' {prob:.1f}%', ha='left', va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        
                        # Statistics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown(f'<div class="stat-card"><b>Top Prediction</b><br/>{ann["label"].title()}</div>', unsafe_allow_html=True)
                        with col_b:
                            st.markdown(f'<div class="stat-card"><b>Confidence</b><br/>{conf:.1f}%</div>', unsafe_allow_html=True)
                        with col_c:
                            runner_up_idx = np.argsort(probs)[-2]
                            st.markdown(f'<div class="stat-card"><b>Runner-up</b><br/>{CLASSES[runner_up_idx].title()} ({probs[runner_up_idx]:.1f}%)</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🔍 How it works
            - Upload a high-quality facial image
            - AI detects facial regions
            - Analyzes skin condition
            """)
        with col2:
            st.markdown("""
            ### ✨ Features
            - Real-time analysis
            - Multi-class detection
            - Confidence metrics
            """)
        with col3:
            st.markdown("""
            ### 📋 Predictions
            - Wrinkles detection
            - Dark spots analysis
            - Puffy eyes detection
            - Clear skin assessment
            """)
        st.info('👋 **Ready to analyze!** Upload a facial image above to get started with skin condition predictions.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception running the app")
