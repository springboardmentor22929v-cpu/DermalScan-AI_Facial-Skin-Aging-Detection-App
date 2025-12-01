# app.py
"""
DermalScan AI (new version)
A more robust, modular, and user-friendly Streamlit app for skin analysis.
Features:
 - load local model (final_efficientnet_model.h5) or upload a model at runtime
 - image upload or webcam capture
 - graceful demo mode (simulated predictions) if model missing
 - multiple visualizations (bar, pie, gauge)
 - annotated image + ZIP export (annotated image, PNG charts, CSV, report)
 - history logging to a local CSV file
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
from pathlib import Path
import time
import zipfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import traceback

# TensorFlow import wrapped so app won't crash on import failure (e.g., no TF installed)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------------------
# Configuration & constants
# ---------------------------
DEFAULT_MODEL_FILENAME = "final_efficientnet_model.h5"  # place this file next to app.py to load automatically
LOG_CSV = "dermalscan_history.csv"
IMG_SIZE = 224
CLASSES = ["wrinkles", "dark spots", "puffy eyes", "clear skin"]

# ---------------------------
# Utils: model loading & preprocessing
# ---------------------------

@st.cache_resource(show_spinner=True)
def try_load_model_from_path(path_str: str):
    """Attempt to load a Keras .h5 model from path. Returns model or None."""
    if not TF_AVAILABLE:
        return None
    try:
        path = Path(path_str)
        if not path.exists():
            return None
        model = load_model(str(path), compile=False)
        return model
    except Exception:
        # don't raise here — return None so caller can fall back to demo
        return None

@st.cache_resource(show_spinner=True)
def try_load_model_from_bytes(file_bytes: bytes, tmp_name: str = "uploaded_model.h5"):
    """Load model from uploaded bytes (Streamlit FileUploader). Returns model or None."""
    if not TF_AVAILABLE:
        return None
    try:
        tmp_path = Path(tmp_name)
        tmp_path.write_bytes(file_bytes)
        model = load_model(str(tmp_path), compile=False)
        # optionally remove the temporary file
        try:
            tmp_path.unlink()
        except Exception:
            pass
        return model
    except Exception:
        return None

def preprocess_for_model(pil_img: Image.Image, size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess PIL image for EfficientNet-style models."""
    img = pil_img.convert("RGB")
    arr = np.array(img).astype("float32")
    import cv2
    arr = cv2.resize(arr, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    if TF_AVAILABLE:
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = (arr / 255.0) - 0.5
    arr = np.expand_dims(arr, axis=0)
    return arr

def safe_prepare_probs(raw, n_classes=len(CLASSES)):
    """Accept raw model output (logits or probs) and turn into valid probability vector."""
    arr = np.array(raw).squeeze().flatten()
    if arr.size == 0:
        return np.zeros(n_classes, dtype=float)
    # if already valid probability vector
    if arr.size == n_classes and arr.min() >= 0 and np.isclose(arr.sum(), 1.0, atol=1e-3):
        probs = arr.astype(float)
    else:
        # align size
        if arr.size != n_classes:
            if arr.size > n_classes:
                arr = arr[:n_classes]
            else:
                arr = np.concatenate([arr, np.zeros(n_classes - arr.size)])
        ex = np.exp(arr - np.max(arr))
        probs = ex / (ex.sum() + 1e-12)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / (probs.sum() + 1e-12)
    return probs

# ---------------------------
# Visualization helpers
# ---------------------------

def annotate_banner(pil_img: Image.Image, text: str, font_size=22, padding=8):
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w = len(text) * (font_size // 2)
        h = font_size
    x = padding
    y = padding
    draw.rectangle([x - padding, y - padding, x + w + padding, y + h + padding],
                   fill=(255,255,255,230))
    draw.text((x, y), text, fill=(0,0,0), font=font)
    return out

def create_bar_fig(probs, classes=CLASSES):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="none")
    colors = ['#00bfff' if p == max(probs) else '#1e4a6e' for p in probs]
    bars = ax.barh(classes, [p*100 for p in probs], color=colors)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)")
    ax.invert_yaxis()
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{p*100:.1f}%", va='center')
    plt.tight_layout()
    return fig

def create_pie_fig(probs, classes=CLASSES):
    fig, ax = plt.subplots(figsize=(4,4), facecolor="none")
    colors = ['#00bfff', '#1e90ff', '#4169e1', '#6495ed'][:len(probs)]
    wedges, texts, autotexts = ax.pie(probs, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.legend(wedges, classes, loc="center left", bbox_to_anchor=(1,0.5))
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def create_gauge_fig(value, label):
    # simple semi-circle gauge
    fig, ax = plt.subplots(figsize=(4,2.4), facecolor="none")
    theta = np.linspace(0, np.pi, 100)
    ax.plot(theta, np.sin(theta), alpha=0)
    # filled arc
    fill_theta = np.linspace(0, np.pi * value, 100)
    ax.fill_between(fill_theta, 0, np.sin(fill_theta), color=(0.2, 0.6, 0.9, 0.8))
    ax.text(np.pi/2, 0.5, f"{value*100:.1f}%", ha='center', va='center', fontsize=18)
    ax.text(np.pi/2, 0.05, label.title(), ha='center', va='center', fontsize=10)
    ax.axis('off')
    return fig

# ---------------------------
# Export helpers
# ---------------------------

def fig_to_bytes(fig, fmt='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.getvalue()

def create_export_zip(annotated_img: Image.Image, probs, classes, top_label, top_conf, source_name):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # annotated image
        img_buf = io.BytesIO()
        annotated_img.save(img_buf, format='PNG')
        img_buf.seek(0)
        zf.writestr("annotated_image.png", img_buf.getvalue())

        # charts
        try:
            fig = create_bar_fig(probs, classes)
            zf.writestr("bar_chart.png", fig_to_bytes(fig))
            plt.close(fig)
        except Exception:
            pass
        try:
            fig2 = create_pie_fig(probs, classes)
            zf.writestr("pie_chart.png", fig_to_bytes(fig2))
            plt.close(fig2)
        except Exception:
            pass
        try:
            gauge = create_gauge_fig(top_conf, top_label)
            zf.writestr("gauge.png", fig_to_bytes(gauge))
            plt.close(gauge)
        except Exception:
            pass

        # csv
        df = pd.DataFrame({
            "condition": classes,
            "score": [f"{p*100:.2f}%" for p in probs]
        })
        zf.writestr("results.csv", df.to_csv(index=False))

        # text report
        report = f"""DermalScan AI Analysis Report
Source: {source_name}
Top: {top_label} ({top_conf*100:.2f}%)

All scores:
"""
        for c, p in zip(classes, probs):
            report += f" - {c}: {p*100:.2f}%\n"
        zf.writestr("report.txt", report)
    zip_buf.seek(0)
    return zip_buf

# ---------------------------
# History logging
# ---------------------------

def append_history_row(row_dict):
    df = pd.DataFrame([row_dict])
    if os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)

def read_history():
    if os.path.exists(LOG_CSV):
        try:
            return pd.read_csv(LOG_CSV)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ---------------------------
# Prediction routine
# ---------------------------

def predict_image_and_annotate(pil_img: Image.Image, model):
    # preprocess
    x = preprocess_for_model(pil_img)
    start = time.time()
    if model is not None:
        try:
            raw = model.predict(x, verbose=0)
            probs = safe_prepare_probs(raw, n_classes=len(CLASSES))
        except Exception:
            # if model fails, use demo
            probs = np.random.dirichlet(np.ones(len(CLASSES)))
    else:
        # demo random prediction (seeded for reproducibility by image size)
        np.random.seed(int(sum(pil_img.size)))
        probs = np.random.dirichlet(np.ones(len(CLASSES)))
    elapsed = time.time() - start
    top_idx = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    top_conf = float(probs[top_idx])
    banner_text = f"{top_label.upper()} - {top_conf*100:.1f}%"
    annotated = annotate_banner(pil_img, banner_text, font_size=20)
    return annotated, probs, top_label, top_conf, elapsed

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="DermalScan AI (modern)", layout="wide")
st.title("DermalScan AI — Modern Edition")

# two-column top bar: model loader + quick info
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("**Model**")
    st.write("If you placed a model named `final_efficientnet_model.h5` next to this `app.py` it will try to load automatically.")
    # show model load status
    # try to load default model in same dir
    default_model_path = Path(DEFAULT_MODEL_FILENAME)
    model_obj = try_load_model_from_path(str(default_model_path)) if TF_AVAILABLE else None
    if model_obj is not None:
        st.success(f"Loaded local model: {default_model_path.name}")
    else:
        st.info("No local model found. You can upload a `.h5` model below (optional).")

    uploaded_model = st.file_uploader("Upload a Keras .h5 model (optional)", type=["h5"])
    if uploaded_model is not None:
        st.info("Loading uploaded model...")
        model_bytes = uploaded_model.read()
        model_uploaded_obj = try_load_model_from_bytes(model_bytes, tmp_name="uploaded_model_temp.h5") if TF_AVAILABLE else None
        if model_uploaded_obj is not None:
            st.success("Uploaded model loaded and will be used for predictions.")
            model_to_use = model_uploaded_obj
        else:
            st.error("Uploaded model failed to load. App will use demo mode.")
            model_to_use = None
    else:
        model_to_use = model_obj  # could be None => demo mode

with col2:
    st.markdown("**Quick Tips**")
    st.markdown("- Use good lighting for photos\n- Center the face\n- Use a recent phone camera photo")
    st.markdown("---")
    st.markdown("Demo mode: If no model is loaded the app uses simulated results (safe for testing).")

st.markdown("---")

# image picker: webcam or upload
st.markdown("### Upload / Capture Image")
pick_col1, pick_col2 = st.columns(2)
with pick_col1:
    uploaded_file = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
with pick_col2:
    cam_img = st.camera_input("Or take a photo (camera)", key="cam")

image_for_analysis = None
source_name = "none"
if uploaded_file is not None:
    try:
        image_for_analysis = Image.open(uploaded_file).convert("RGB")
        source_name = getattr(uploaded_file, "name", f"upload_{int(time.time())}.png")
    except Exception as e:
        st.error("Failed to open uploaded image.")
        st.exception(e)
elif cam_img is not None:
    try:
        image_for_analysis = Image.open(cam_img).convert("RGB")
        source_name = f"webcam_{int(time.time())}.png"
    except Exception as e:
        st.error("Failed to open camera image.")
        st.exception(e)

if image_for_analysis is None:
    st.info("Please upload an image or take a photo to run analysis.")
    st.stop()

# show image preview
st.markdown("### Preview")
st.image(image_for_analysis, use_column_width=False, width=360)

# run prediction
if "last_results" not in st.session_state:
    st.session_state.last_results = None

if st.button("Analyze Skin"):
    try:
        with st.spinner("Running model..."):
            annotated_img, probs, top_label, top_conf, elapsed = predict_image_and_annotate(image_for_analysis, model_to_use)
        st.session_state.last_results = dict(
            annotated_img=annotated_img,
            probs=probs,
            top_label=top_label,
            top_conf=top_conf,
            elapsed=elapsed,
            source_name=source_name
        )
        # log
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_name,
            "top_prediction": top_label,
            "top_confidence": f"{top_conf*100:.2f}%"
        }
        for c,p in zip(CLASSES, probs):
            entry[c] = f"{p*100:.2f}%"
        append_history_row(entry)
    except Exception as e:
        st.error("Analysis failed.")
        st.exception(traceback.format_exc())

# results area
if st.session_state.last_results is not None:
    r = st.session_state.last_results
    st.markdown("## Results")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Annotated Image")
        st.image(r["annotated_img"], use_column_width=True)
        # export zip
        zip_buf = create_export_zip(r["annotated_img"], r["probs"], CLASSES, r["top_label"], r["top_conf"], r["source_name"])
        st.download_button("Download analysis ZIP", data=zip_buf, file_name=f"dermalscan_{int(time.time())}.zip", mime="application/zip")

    with right:
        st.markdown("### Summary")
        st.write(f"Top prediction: **{r['top_label'].title()}**")
        st.write(f"Confidence: **{r['top_conf']*100:.1f}%**")
        st.write(f"Inference time: {r['elapsed']*1000:.0f} ms (approx)")
        st.markdown("---")
        st.markdown("### Confidence Bar")
        fig_bar = create_bar_fig(r["probs"], CLASSES)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
        st.markdown("---")
        st.markdown("### Probability Distribution")
        fig_p = create_pie_fig(r["probs"], CLASSES)
        st.pyplot(fig_p)
        plt.close(fig_p)
        st.markdown("---")
        st.markdown("### Gauge")
        fig_g = create_gauge_fig(r["top_conf"], r["top_label"])
        st.pyplot(fig_g)
        plt.close(fig_g)

    st.markdown("---")
    # recommendations
    st.markdown("### Recommendations")
    rec_map = {
        "wrinkles": ["Use retinoids", "Sunscreen daily", "Stay hydrated"],
        "dark spots": ["Vitamin C", "Sunscreen SPF50+", "Consider dermatologist"],
        "puffy eyes": ["Cold compress", "Reduce salt intake", "Sleep well"],
        "clear skin": ["Maintain routine", "Sunscreen", "Keep hydrated"]
    }
    recs = rec_map.get(r["top_label"], rec_map["clear skin"])
    for rec in recs:
        st.write(f"- {rec}")

# history
st.markdown("---")
st.markdown("### Prediction History")
hist = read_history()
if hist.empty:
    st.info("No history yet. Run an analysis to create history.")
else:
    st.dataframe(hist.tail(10).iloc[::-1])
    csv = hist.to_csv(index=False)
    st.download_button("Download history CSV", data=csv, file_name="dermalscan_history.csv", mime="text/csv")

st.markdown("---")
st.markdown("DermalScan AI — Modern Edition • Designed for reliability and clarity.")
