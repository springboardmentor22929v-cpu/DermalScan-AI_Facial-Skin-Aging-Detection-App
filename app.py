"""
DermalScan AI - Advanced Skin Analysis Platform
Predicts: wrinkles, dark spots, puffy eyes, clear skin
Enhanced UI with animated dark spots background and unified export feature
"""
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io
import os
import time
import base64
import zipfile
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

CLASSES = ['wrinkles', 'dark spots', 'puffy eyes', 'clear skin']
IMG_SIZE = 224
MODEL_PATH = r'C:\Users\tanni\Downloads\final_efficientnet_model1.h5'
LOG_CSV = "dermalscan_prediction_log.csv"

@st.cache_resource(show_spinner=False)
def load_model_auto(path):
    try:
        m = load_model(path, compile=False)
        return m
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_hero_image(local_path=None):
    try:
        if local_path and os.path.exists(local_path):
            return Image.open(local_path)
        url = "https://img.freepik.com/free-photo/facial-recognition-collage-concept_23-2150038888.jpg"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except:
        pass
    return None

def preprocess_image(pil_img: Image.Image, size=(IMG_SIZE, IMG_SIZE)):
    img = pil_img.convert("RGB")
    arr = np.array(img).astype("float32")
    arr = cv2.resize(arr, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    arr = np.expand_dims(arr, axis=0)
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    return preprocess_input(arr)

def safe_prepare_probs(raw, n_classes=len(CLASSES)):
    arr = np.array(raw).squeeze().flatten()
    if arr.size == 0:
        return np.zeros(n_classes, dtype=float)
    if arr.size == n_classes and arr.min() >= 0 and np.isclose(arr.sum(), 1.0, atol=1e-3):
        probs = arr.astype(float)
    else:
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

def annotate_banner(pil_img: Image.Image, text: str, banner_h=80, font_size=20):
    out = pil_img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width = len(text) * 10
        text_height = font_size

    x = 10
    y = 10
    padding = 8

    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(255, 255, 255),
        outline=(200, 200, 200),
        width=2
    )

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return out

def create_confidence_bar_chart(probs, classes):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    colors = ['#00bfff' if p == max(probs) else '#1e4a6e' for p in probs]
    bars = ax.barh(classes, [p * 100 for p in probs], color=colors, edgecolor='#00bfff', linewidth=1)
    
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', ha='left', 
                color='#87ceeb', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Confidence (%)', color='#87ceeb', fontsize=12, fontweight='bold')
    ax.set_title('Skin Condition Analysis Results', color='#e8f4f8', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 110)
    ax.tick_params(axis='both', colors='#87ceeb', labelsize=11)
    ax.spines['bottom'].set_color('#00bfff')
    ax.spines['left'].set_color('#00bfff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#1e4a6e', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_pie_chart(probs, classes):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a1628')
    
    colors = ['#00bfff', '#1e90ff', '#4169e1', '#6495ed']
    explode = [0.08 if p == max(probs) else 0.02 for p in probs]
    
    wedges, texts, autotexts = ax.pie(
        probs, 
        labels=None,
        autopct='%1.1f%%',
        explode=explode,
        colors=colors,
        startangle=90,
        textprops={'color': '#ffffff', 'fontsize': 14, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': '#0a1628', 'linewidth': 3},
        pctdistance=0.75
    )
    
    for autotext in autotexts:
        autotext.set_color('#ffffff')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(13)
    
    ax.legend(
        wedges, 
        [f'{cls.title()}' for cls in classes],
        title="Conditions",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=12,
        title_fontsize=14,
        frameon=True,
        facecolor='#0a1628',
        edgecolor='#00bfff',
        labelcolor='#e8f4f8'
    )
    
    ax.set_title('Probability Distribution', color='#e8f4f8', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_gauge_chart(confidence, label):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    ax.fill_between(theta, 0, r, alpha=0.2, color='#1e4a6e')
    
    fill_theta = np.linspace(0, np.pi * confidence, 100)
    color = '#00ff00' if confidence > 0.7 else '#ffff00' if confidence > 0.4 else '#ff6347'
    ax.fill_between(fill_theta, 0.7, r, alpha=0.8, color=color)
    
    ax.set_xlim(-0.1, np.pi + 0.1)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    
    ax.text(np.pi/2, 0.3, f'{confidence*100:.1f}%', ha='center', va='center', 
            fontsize=24, color='#e8f4f8', fontweight='bold')
    ax.text(np.pi/2, 0.05, label.upper(), ha='center', va='center', 
            fontsize=12, color='#87ceeb', fontweight='bold')
    
    ax.set_title('Confidence Score', color='#e8f4f8', fontsize=14, fontweight='bold', y=1.05)
    
    return fig

def create_full_prediction_chart(probs, classes, top_label, top_conf):
    fig = plt.figure(figsize=(14, 10), facecolor='#0a1628')
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#0a1628')
    colors = ['#00bfff' if p == max(probs) else '#1e4a6e' for p in probs]
    bars = ax1.barh(classes, [p * 100 for p in probs], color=colors, edgecolor='#00bfff', linewidth=1)
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', ha='left', 
                color='#87ceeb', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Confidence (%)', color='#87ceeb', fontsize=10, fontweight='bold')
    ax1.set_title('Confidence Bar Chart', color='#e8f4f8', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.tick_params(axis='both', colors='#87ceeb', labelsize=9)
    ax1.spines['bottom'].set_color('#00bfff')
    ax1.spines['left'].set_color('#00bfff')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(2, 2, 2)
    pie_colors = ['#00bfff', '#1e90ff', '#4169e1', '#6495ed']
    explode = [0.05 if p == max(probs) else 0 for p in probs]
    wedges, texts, autotexts = ax2.pie(
        probs, 
        labels=None,
        autopct='%1.1f%%',
        explode=explode,
        colors=pie_colors,
        startangle=90,
        textprops={'color': '#ffffff', 'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': '#0a1628', 'linewidth': 2}
    )
    ax2.legend(wedges, [c.title() for c in classes], loc='upper right', fontsize=9,
               facecolor='#0a1628', edgecolor='#00bfff', labelcolor='#e8f4f8')
    ax2.set_title('Probability Distribution', color='#e8f4f8', fontsize=12, fontweight='bold')
    
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#0a1628')
    theta = np.linspace(0, np.pi, 100)
    ax3.fill_between(theta, 0, 1, alpha=0.2, color='#1e4a6e')
    fill_theta = np.linspace(0, np.pi * top_conf, 100)
    gauge_color = '#00ff00' if top_conf > 0.7 else '#ffff00' if top_conf > 0.4 else '#ff6347'
    ax3.fill_between(fill_theta, 0.7, 1, alpha=0.8, color=gauge_color)
    ax3.set_xlim(-0.1, np.pi + 0.1)
    ax3.set_ylim(0, 1.2)
    ax3.axis('off')
    ax3.text(np.pi/2, 0.3, f'{top_conf*100:.1f}%', ha='center', va='center', 
            fontsize=20, color='#e8f4f8', fontweight='bold')
    ax3.text(np.pi/2, 0.05, top_label.upper(), ha='center', va='center', 
            fontsize=10, color='#87ceeb', fontweight='bold')
    ax3.set_title('Top Prediction Gauge', color='#e8f4f8', fontsize=12, fontweight='bold')
    
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#0a1628')
    ax4.axis('off')
    summary_text = f"""
ANALYSIS SUMMARY
================

Top Prediction: {top_label.title()}
Confidence: {top_conf*100:.2f}%

All Predictions:
"""
    for cls, prob in zip(classes, probs):
        summary_text += f"  • {cls.title()}: {prob*100:.2f}%\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', color='#e8f4f8', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1e4a6e', alpha=0.5, edgecolor='#00bfff'))
    ax4.set_title('Analysis Summary', color='#e8f4f8', fontsize=12, fontweight='bold')
    
    fig.suptitle('DermalScan AI - Full Prediction Analysis', color='#00bfff', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def fig_to_bytes(fig, fmt='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    return buf.getvalue()

def create_export_zip(annotated_img, probs, classes, top_label, top_conf, source_name):
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        img_buffer = io.BytesIO()
        annotated_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        zip_file.writestr('annotated_image.png', img_buffer.getvalue())
        
        try:
            bar_chart = create_confidence_bar_chart(probs, classes)
            bar_bytes = fig_to_bytes(bar_chart)
            zip_file.writestr('confidence_bar_chart.png', bar_bytes)
        finally:
            plt.close('all')
        
        try:
            pie_chart = create_pie_chart(probs, classes)
            pie_bytes = fig_to_bytes(pie_chart)
            zip_file.writestr('probability_distribution.png', pie_bytes)
        finally:
            plt.close('all')
        
        try:
            gauge_chart = create_gauge_chart(top_conf, top_label)
            gauge_bytes = fig_to_bytes(gauge_chart)
            zip_file.writestr('confidence_gauge.png', gauge_bytes)
        finally:
            plt.close('all')
        
        try:
            full_chart = create_full_prediction_chart(probs, classes, top_label, top_conf)
            full_bytes = fig_to_bytes(full_chart)
            zip_file.writestr('full_prediction_analysis.png', full_bytes)
        finally:
            plt.close('all')
        
        results_data = {
            'Analysis Report': ['DermalScan AI Analysis Results'],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Source': [source_name],
            'Top Prediction': [top_label],
            'Confidence': [f'{top_conf*100:.2f}%']
        }
        for cls, prob in zip(classes, probs):
            results_data[f'{cls.title()} Score'] = [f'{prob*100:.2f}%']
        
        df = pd.DataFrame(results_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('analysis_results.csv', csv_buffer.getvalue())
        
        report_text = f"""
================================================================================
                        DERMALSCAN AI ANALYSIS REPORT
================================================================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Image: {source_name}

--------------------------------------------------------------------------------
                              PRIMARY DIAGNOSIS
--------------------------------------------------------------------------------

Detected Condition: {top_label.upper()}
Confidence Level: {top_conf*100:.2f}%

--------------------------------------------------------------------------------
                           DETAILED PROBABILITY SCORES
--------------------------------------------------------------------------------

"""
        for cls, prob in zip(classes, probs):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            report_text += f"  {cls.title():15} [{bar}] {prob*100:6.2f}%\n"
        
        report_text += f"""
--------------------------------------------------------------------------------
                              RECOMMENDATIONS
--------------------------------------------------------------------------------

Based on the analysis results:

"""
        if top_label == 'wrinkles':
            report_text += """  • Consider using retinoid-based skincare products
  • Apply sunscreen daily to prevent further damage
  • Stay hydrated and maintain a healthy diet
  • Consider consulting a dermatologist for personalized treatment
"""
        elif top_label == 'dark spots':
            report_text += """  • Use products containing Vitamin C or Niacinamide
  • Apply broad-spectrum sunscreen religiously
  • Consider chemical exfoliants like AHA/BHA
  • Consult a dermatologist for professional treatments
"""
        elif top_label == 'puffy eyes':
            report_text += """  • Ensure adequate sleep (7-9 hours)
  • Reduce salt intake to minimize water retention
  • Apply cold compresses or chilled eye masks
  • Use caffeine-infused eye creams
"""
        else:
            report_text += """  • Continue your current skincare routine
  • Maintain sun protection habits
  • Stay hydrated and eat a balanced diet
  • Regular check-ups to maintain skin health
"""
        
        report_text += """
================================================================================
                    Generated by DermalScan AI Analysis Platform
================================================================================
"""
        zip_file.writestr('detailed_report.txt', report_text)
    
    zip_buffer.seek(0)
    return zip_buffer

def log_prediction(entry: dict):
    df = pd.DataFrame([entry])
    if os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_CSV, index=False)

def get_prediction_history():
    if os.path.exists(LOG_CSV):
        try:
            df = pd.read_csv(LOG_CSV)
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

st.set_page_config(page_title="DermalScan AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html {
        scroll-behavior: smooth;
    }

    .stApp > header {
        background-color: transparent;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0a1628 0%, #0d1b2a 50%, #0a1628 100%);
        padding: 0;
        position: relative;
        overflow-x: hidden;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    .dark-spots-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .dark-spot {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle at center, rgba(0, 191, 255, 0.4), rgba(30, 144, 255, 0.2), transparent);
        pointer-events: none;
        animation: floatSpot linear infinite;
    }

    @keyframes floatSpot {
        0% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 0.6;
        }
        90% {
            opacity: 0.6;
        }
        100% {
            transform: translateY(-10vh) rotate(360deg);
            opacity: 0;
        }
    }

    .dark-spot:nth-child(1) { width: 4px; height: 4px; left: 5%; animation-duration: 25s; animation-delay: 0s; }
    .dark-spot:nth-child(2) { width: 6px; height: 6px; left: 10%; animation-duration: 20s; animation-delay: 2s; }
    .dark-spot:nth-child(3) { width: 3px; height: 3px; left: 15%; animation-duration: 28s; animation-delay: 4s; }
    .dark-spot:nth-child(4) { width: 5px; height: 5px; left: 20%; animation-duration: 22s; animation-delay: 1s; }
    .dark-spot:nth-child(5) { width: 4px; height: 4px; left: 25%; animation-duration: 26s; animation-delay: 3s; }
    .dark-spot:nth-child(6) { width: 7px; height: 7px; left: 30%; animation-duration: 18s; animation-delay: 5s; }
    .dark-spot:nth-child(7) { width: 3px; height: 3px; left: 35%; animation-duration: 24s; animation-delay: 0s; }
    .dark-spot:nth-child(8) { width: 5px; height: 5px; left: 40%; animation-duration: 21s; animation-delay: 2s; }
    .dark-spot:nth-child(9) { width: 4px; height: 4px; left: 45%; animation-duration: 27s; animation-delay: 4s; }
    .dark-spot:nth-child(10) { width: 6px; height: 6px; left: 50%; animation-duration: 19s; animation-delay: 1s; }
    .dark-spot:nth-child(11) { width: 3px; height: 3px; left: 55%; animation-duration: 23s; animation-delay: 3s; }
    .dark-spot:nth-child(12) { width: 5px; height: 5px; left: 60%; animation-duration: 25s; animation-delay: 5s; }
    .dark-spot:nth-child(13) { width: 4px; height: 4px; left: 65%; animation-duration: 20s; animation-delay: 0s; }
    .dark-spot:nth-child(14) { width: 7px; height: 7px; left: 70%; animation-duration: 22s; animation-delay: 2s; }
    .dark-spot:nth-child(15) { width: 3px; height: 3px; left: 75%; animation-duration: 28s; animation-delay: 4s; }
    .dark-spot:nth-child(16) { width: 5px; height: 5px; left: 80%; animation-duration: 24s; animation-delay: 1s; }
    .dark-spot:nth-child(17) { width: 4px; height: 4px; left: 85%; animation-duration: 26s; animation-delay: 3s; }
    .dark-spot:nth-child(18) { width: 6px; height: 6px; left: 90%; animation-duration: 18s; animation-delay: 5s; }
    .dark-spot:nth-child(19) { width: 3px; height: 3px; left: 95%; animation-duration: 21s; animation-delay: 0s; }
    .dark-spot:nth-child(20) { width: 5px; height: 5px; left: 2%; animation-duration: 27s; animation-delay: 2s; }
    .dark-spot:nth-child(21) { width: 4px; height: 4px; left: 8%; animation-duration: 19s; animation-delay: 4s; }
    .dark-spot:nth-child(22) { width: 6px; height: 6px; left: 12%; animation-duration: 23s; animation-delay: 1s; }
    .dark-spot:nth-child(23) { width: 3px; height: 3px; left: 18%; animation-duration: 25s; animation-delay: 3s; }
    .dark-spot:nth-child(24) { width: 5px; height: 5px; left: 22%; animation-duration: 20s; animation-delay: 5s; }
    .dark-spot:nth-child(25) { width: 4px; height: 4px; left: 28%; animation-duration: 22s; animation-delay: 0s; }
    .dark-spot:nth-child(26) { width: 7px; height: 7px; left: 32%; animation-duration: 28s; animation-delay: 2s; }
    .dark-spot:nth-child(27) { width: 3px; height: 3px; left: 38%; animation-duration: 24s; animation-delay: 4s; }
    .dark-spot:nth-child(28) { width: 5px; height: 5px; left: 42%; animation-duration: 26s; animation-delay: 1s; }
    .dark-spot:nth-child(29) { width: 4px; height: 4px; left: 48%; animation-duration: 18s; animation-delay: 3s; }
    .dark-spot:nth-child(30) { width: 6px; height: 6px; left: 52%; animation-duration: 21s; animation-delay: 5s; }

    .hero-section {
        padding: 2rem 2rem 2rem 2rem;
        position: relative;
        z-index: 1;
        border-bottom: 1px solid rgba(0, 191, 255, 0.1);
    }

    .hero-content {
        display: flex;
        align-items: center;
        gap: 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    .hero-image-container {
        flex: 1;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .hero-image-container img {
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0, 191, 255, 0.2);
        border: 2px solid rgba(0, 191, 255, 0.3);
        max-width: 100%;
        height: auto;
        transition: all 0.3s ease;
    }

    .hero-image-container img:hover {
        box-shadow: 0 25px 70px rgba(0, 191, 255, 0.3);
        transform: scale(1.02);
    }

    .hero-text {
        flex: 1;
        text-align: left;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #e8f4f8;
        margin: 0 0 1rem 0;
        letter-spacing: -1px;
        line-height: 1.2;
    }

    .hero-title-accent {
        background: linear-gradient(135deg, #00bfff 0%, #1e90ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        color: #87ceeb;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
        max-width: 600px;
        line-height: 1.7;
        opacity: 0.9;
    }

    .start-button {
        display: inline-block;
        background: linear-gradient(135deg, #00bfff 0%, #0d8ec9 100%);
        color: white !important;
        font-weight: 700;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 191, 255, 0.3);
        font-size: 1.1rem;
        box-shadow: 0 8px 20px rgba(0, 191, 255, 0.2);
        transition: all 0.3s ease;
        text-decoration: none;
        cursor: pointer;
        letter-spacing: 0.5px;
    }

    .start-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 191, 255, 0.4);
        background: linear-gradient(135deg, #1e90ff 0%, #00bfff 100%);
    }

    @media (max-width: 968px) {
        .hero-content {
            flex-direction: column;
            gap: 2rem;
        }
        .hero-title {
            font-size: 2rem;
        }
    }

    .content-wrapper {
        padding: 1.5rem 2rem;
        position: relative;
        z-index: 1;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e8f4f8;
        margin: 1rem 0 0.8rem 0;
        letter-spacing: 0.3px;
        opacity: 0.95;
    }

    .section-subtitle {
        font-size: 0.9rem;
        color: #87ceeb;
        font-weight: 500;
        margin-top: 0.2rem;
        opacity: 0.8;
    }

    .feature-card {
        background: rgba(10, 22, 40, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 191, 255, 0.15);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    .feature-card:hover {
        border-color: rgba(0, 191, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 191, 255, 0.3), 0 0 60px rgba(0, 191, 255, 0.15);
        transform: translateY(-5px);
        background: rgba(10, 22, 40, 0.8);
    }

    .input-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #87ceeb;
        margin-bottom: 0.5rem;
        letter-spacing: 0.3px;
    }

    .input-hint {
        color: #6b9eb8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
        opacity: 0.8;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00bfff 0%, #0d8ec9 100%);
        color: white;
        font-weight: 700;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 191, 255, 0.3);
        font-size: 1rem;
        box-shadow: 0 8px 20px rgba(0, 191, 255, 0.2);
        transition: all 0.3s ease;
        width: auto;
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0, 191, 255, 0.3);
        background: linear-gradient(135deg, #1e90ff 0%, #00bfff 100%);
        border-color: rgba(0, 191, 255, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 191, 255, 0.15);
        transition: all 0.4s ease;
    }

    div[data-testid="stImage"]:hover {
        box-shadow: 0 0 25px rgba(0, 191, 255, 0.3), 0 15px 40px rgba(0, 0, 0, 0.3);
        transform: translateY(-3px);
        border-color: rgba(0, 191, 255, 0.5);
    }

    .stats-box {
        background: rgba(10, 22, 40, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.3rem 0;
        border: 2px solid rgba(0, 191, 255, 0.5);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2), 0 0 15px rgba(0, 191, 255, 0.15);
        transition: all 0.4s ease;
    }

    .stats-box:hover {
        background: rgba(10, 22, 40, 0.9);
        box-shadow: 0 0 25px rgba(0, 191, 255, 0.35), 0 8px 24px rgba(0, 0, 0, 0.3);
        transform: translateY(-3px);
        border-color: #00ff88;
    }

    .stats-label {
        color: #87ceeb;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.9;
    }

    .stats-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    .result-card {
        background: rgba(10, 22, 40, 0.7);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 191, 255, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s ease;
    }

    .result-card:hover {
        box-shadow: 0 0 30px rgba(0, 191, 255, 0.2), 0 12px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(0, 191, 255, 0.5);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #00bfff 0%, #0d8ec9 100%);
        color: white;
        font-weight: 700;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 191, 255, 0.3);
        box-shadow: 0 6px 18px rgba(0, 191, 255, 0.2);
        transition: all 0.3s ease;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.35), 0 10px 25px rgba(0, 191, 255, 0.25);
        background: linear-gradient(135deg, #1e90ff 0%, #00bfff 100%);
    }

    .small-download-btn .stDownloadButton > button {
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
    }

    .info-box {
        background: rgba(0, 191, 255, 0.08);
        backdrop-filter: blur(10px);
        border-left: 3px solid #1e90ff;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        color: #87ceeb;
        margin: 0.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.4s ease;
    }

    .info-box:hover {
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.2);
        border-left-color: #00bfff;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(0, 191, 255, 0.3), 
            transparent);
        margin: 1.5rem 0;
    }

    [data-testid="stFileUploader"] {
        background: rgba(10, 22, 40, 0.4);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(0, 191, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.4s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 191, 255, 0.7);
        background: rgba(0, 191, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.2);
    }

    [data-testid="stCameraInput"] {
        background: rgba(10, 22, 40, 0.4);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(0, 191, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.4s ease;
    }

    [data-testid="stCameraInput"]:hover {
        border-color: rgba(0, 191, 255, 0.7);
        background: rgba(0, 191, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.2);
    }

    .stSpinner > div {
        border-color: #00bfff !important;
        border-right-color: transparent !important;
    }

    .recommendation-card {
        background: linear-gradient(135deg, rgba(0, 191, 255, 0.1), rgba(30, 144, 255, 0.05));
        border: 1px solid rgba(0, 191, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.4s ease;
    }

    .recommendation-card:hover {
        box-shadow: 0 0 25px rgba(0, 191, 255, 0.25);
        border-color: rgba(0, 191, 255, 0.5);
        transform: translateY(-3px);
    }

    .recommendation-title {
        color: #00bfff;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .recommendation-text {
        color: #87ceeb;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    .confidence-meter {
        background: rgba(10, 22, 40, 0.8);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(0, 191, 255, 0.2);
        transition: all 0.4s ease;
    }

    .confidence-meter:hover {
        box-shadow: 0 0 20px rgba(0, 191, 255, 0.2);
        border-color: rgba(0, 191, 255, 0.5);
    }

    .confidence-bar-container {
        background: rgba(30, 74, 110, 0.3);
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        margin: 0.3rem 0;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00bfff, #1e90ff);
        transition: width 0.5s ease;
    }

    .history-table {
        background: rgba(10, 22, 40, 0.6);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(0, 191, 255, 0.2);
    }

    .stDataFrame {
        background: rgba(10, 22, 40, 0.4) !important;
    }

    .stDataFrame table {
        color: #e8f4f8 !important;
    }

    .stDataFrame th {
        background: rgba(0, 191, 255, 0.2) !important;
        color: #00bfff !important;
    }

    .stDataFrame td {
        color: #87ceeb !important;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.6rem;
        }
        .section-title {
            font-size: 1rem;
        }
        .content-wrapper {
            padding: 0.5rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

spots_html = '<div class="dark-spots-container">'
for i in range(30):
    spots_html += '<div class="dark-spot"></div>'
spots_html += '</div>'
st.markdown(spots_html, unsafe_allow_html=True)

with st.spinner("Loading AI model..."):
    model = load_model_auto(MODEL_PATH)

if model is None:
    st.warning(
        "Model file not found. Running in demo mode with simulated predictions. "
        "Please upload the model file to the correct path for actual predictions."
    )

hero_image = load_hero_image(r"C:\Users\tanni\Downloads\2150038888.jpg")

st.markdown("<div class='hero-section' id='hero'>", unsafe_allow_html=True)

hero_col1, hero_col2 = st.columns([1.2, 1], gap="large")

with hero_col1:
    st.markdown(
        """
        <div class='hero-text'>
            <h1 class='hero-title'>
                <span class='hero-title-accent'>DermalScan</span> — AI Facial Skin Detection App
            </h1>
            <p class='hero-subtitle'>
                Unlock deeper insights into your skin with DermalScan — a next-generation AI platform designed to analyze facial features with remarkable accuracy. DermalScan uses advanced deep-learning models and computer-vision techniques to evaluate your skin in real time and provide a complete assessment of visible skin concerns. The system intelligently scans your face to identify key indicators such as wrinkles, dark spots, clear skin, puffy eyes, and overall skin clarity.
            </p>
            <a href='#features' class='start-button'>Start Now →</a>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_col2:
    if hero_image is not None:
        st.image(hero_image, use_container_width=True)
    else:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, rgba(0, 191, 255, 0.2), rgba(30, 144, 255, 0.1)); 
                        border-radius: 16px; padding: 4rem 2rem; text-align: center; 
                        border: 2px dashed rgba(0, 191, 255, 0.3);'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🔬</div>
                <div style='color: #87ceeb; font-size: 1rem;'>Facial Recognition Concept</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='content-wrapper' id='features'>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='section-title'>Upload or Capture Your Image</div>
    <div class='section-subtitle'>Choose your preferred method to analyze your skin</div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("<div class='input-label'>Webcam Capture</div>", unsafe_allow_html=True)
    cam_bytes = st.camera_input("Take a photo", key="camera_input", label_visibility="collapsed")
    st.markdown("<div class='input-hint'>Capture a live photo</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-label'>File Upload</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="file_input", label_visibility="collapsed")
    st.markdown("<div class='input-hint'>Drag & drop or browse</div>", unsafe_allow_html=True)

uploaded_img = None
source_name = None

if uploaded_file is not None:
    try:
        uploaded_img = Image.open(uploaded_file).convert("RGB")
        source_name = getattr(uploaded_file, "name", f"upload_{int(time.time())}.png")
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
elif cam_bytes is not None:
    try:
        uploaded_img = Image.open(cam_bytes).convert("RGB")
        source_name = f"webcam_{int(time.time())}.png"
    except Exception as e:
        st.error(f"Could not open camera capture: {e}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='section-title'>Analysis Dashboard</div>
    <div class='section-subtitle'>View your original image and AI-powered analysis results</div>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("**Original Image**")
    if uploaded_img is not None:
        st.image(uploaded_img, use_container_width=True)
    else:
        st.markdown("<div class='info-box'>Upload or capture an image to begin your skin analysis</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("**Analysis Result**")
    annotated_slot = st.empty()

btn_col1, btn_col2 = st.columns([1, 1], gap="small")
with btn_col1:
    predict_clicked = st.button("Analyze Skin Condition", key="predict_button", type="primary")
with btn_col2:
    export_slot = st.empty()

def predict_image_and_annotate(pil_img: Image.Image):
    x = preprocess_image(pil_img)
    t0 = time.time()
    
    if model is not None:
        raw = model.predict(x, verbose=0)
        probs = safe_prepare_probs(raw)
    else:
        np.random.seed(hash(str(pil_img.size)) % 2**32)
        probs = np.random.dirichlet(np.ones(len(CLASSES)))
    
    elapsed = time.time() - t0
    top_idx = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    top_conf = float(probs[top_idx])
    banner_text = f"{top_label.upper()} - {top_conf*100:.1f}%"
    annotated = annotate_banner(pil_img, banner_text, banner_h=64, font_size=28)
    return annotated, probs, top_label, top_conf, elapsed

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

if predict_clicked:
    if uploaded_img is None:
        st.warning("Please upload an image or capture from webcam before analyzing.")
    else:
        with st.spinner("Running AI analysis..."):
            try:
                annotated_img, probs, top_label, top_conf, elapsed = predict_image_and_annotate(uploaded_img)
                
                st.session_state.prediction_results = {
                    'annotated_img': annotated_img,
                    'probs': probs,
                    'top_label': top_label,
                    'top_conf': top_conf,
                    'elapsed': elapsed,
                    'source_name': source_name
                }
                
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': source_name,
                    'prediction': top_label,
                    'confidence': f'{top_conf*100:.2f}%',
                    **{cls: f'{p*100:.2f}%' for cls, p in zip(CLASSES, probs)}
                }
                log_prediction(log_entry)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    annotated_img = results['annotated_img']
    probs = results['probs']
    top_label = results['top_label']
    top_conf = results['top_conf']
    elapsed = results['elapsed']
    source_name = results['source_name']
    
    annotated_slot.image(annotated_img, use_container_width=True)
    
    zip_buffer = create_export_zip(annotated_img, probs, CLASSES, top_label, top_conf, source_name)
    export_slot.download_button(
        "Export Results",
        data=zip_buffer,
        file_name=f"dermalscan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        key="export_button"
    )
    
    stats_col1, stats_col2 = st.columns(2, gap="small")
    with stats_col1:
        st.markdown(
            f"""
            <div class='stats-box'>
                <div class='stats-label'>Top Prediction</div>
                <div class='stats-value'>{top_label.title()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with stats_col2:
        st.markdown(
            f"""
            <div class='stats-box'>
                <div class='stats-label'>Confidence</div>
                <div class='stats-value'>{top_conf*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='section-title'>Detailed Analysis</div>
        <div class='section-subtitle'>Confidence scores and probability distribution</div>
        """,
        unsafe_allow_html=True
    )
    
    chart_col1, chart_col2 = st.columns([1.2, 1], gap="medium")
    
    with chart_col1:
        st.markdown("**Confidence Scores**")
        bar_chart = create_confidence_bar_chart(probs, CLASSES)
        st.pyplot(bar_chart, clear_figure=True)
        plt.close('all')
    
    with chart_col2:
        st.markdown("**Probability Distribution**")
        pie_chart = create_pie_chart(probs, CLASSES)
        st.pyplot(pie_chart, clear_figure=True)
        plt.close('all')

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='section-title'>Personalized Recommendations</div>
        <div class='section-subtitle'>Based on your skin analysis results</div>
        """,
        unsafe_allow_html=True
    )
    
    recommendations = {
        'wrinkles': [
            ("Skincare", "Consider using retinoid-based products to boost collagen production and reduce fine lines."),
            ("Sun Protection", "Apply broad-spectrum SPF 30+ sunscreen daily to prevent further UV damage."),
            ("Hydration", "Use hyaluronic acid serums to maintain skin moisture and plumpness."),
            ("Lifestyle", "Stay hydrated, get adequate sleep, and include antioxidant-rich foods in your diet.")
        ],
        'dark spots': [
            ("Brightening", "Use products with Vitamin C, Niacinamide, or Arbutin to fade pigmentation."),
            ("Exfoliation", "Incorporate gentle AHA/BHA exfoliants to promote cell turnover."),
            ("Sun Protection", "Wear SPF 50+ daily - UV exposure worsens dark spots significantly."),
            ("Professional", "Consider consulting a dermatologist for treatments like chemical peels or laser therapy.")
        ],
        'puffy eyes': [
            ("Sleep", "Ensure 7-9 hours of quality sleep nightly and elevate your head slightly."),
            ("Cold Therapy", "Apply chilled spoons, cucumber slices, or cold eye masks to reduce swelling."),
            ("Eye Care", "Use caffeine-infused eye creams to constrict blood vessels and reduce puffiness."),
            ("Diet", "Reduce salt intake and stay hydrated to minimize water retention.")
        ],
        'clear skin': [
            ("Maintenance", "Continue your current skincare routine - it's clearly working well!"),
            ("Prevention", "Maintain consistent sun protection to preserve your healthy skin."),
            ("Hydration", "Keep skin hydrated with appropriate moisturizers for your skin type."),
            ("Check-ups", "Schedule regular dermatology visits to maintain optimal skin health.")
        ]
    }
    
    recs = recommendations.get(top_label, recommendations['clear skin'])
    
    rec_cols = st.columns(2, gap="medium")
    for i, (title, text) in enumerate(recs):
        with rec_cols[i % 2]:
            st.markdown(
                f"""
                <div class='recommendation-card'>
                    <div class='recommendation-title'>{title}</div>
                    <div class='recommendation-text'>{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='section-title'>All Confidence Scores</div>
        <div class='section-subtitle'>Detailed breakdown of prediction probabilities</div>
        """,
        unsafe_allow_html=True
    )
    
    for cls, prob in zip(CLASSES, probs):
        color = '#00ff88' if cls == top_label else '#00bfff'
        st.markdown(
            f"""
            <div class='confidence-meter'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                    <span style='color: #e8f4f8; font-weight: 600;'>{cls.title()}</span>
                    <span style='color: {color}; font-weight: 700;'>{prob*100:.1f}%</span>
                </div>
                <div class='confidence-bar-container'>
                    <div class='confidence-bar' style='width: {prob*100}%; background: linear-gradient(90deg, {color}, #1e90ff);'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='section-title'>Prediction History</div>
    <div class='section-subtitle'>View and download your past analysis records</div>
    """,
    unsafe_allow_html=True
)

history_df = get_prediction_history()

if not history_df.empty:
    st.dataframe(
        history_df.tail(10).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = history_df.to_csv(index=False)
    st.markdown("<div class='small-download-btn'>", unsafe_allow_html=True)
    st.download_button(
        "Download History",
        data=csv_data,
        file_name=f"dermalscan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="history_download_button"
    )
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='info-box'>No prediction history available yet. Analyze an image to start building your history.</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; padding: 1rem; color: #6b9eb8; font-size: 0.85rem;'>
        DermalScan AI - Advanced Skin Analysis Platform<br/>
        <span style='font-size: 0.75rem; opacity: 0.7;'>Powered by Deep Learning Technology</span>
    </div>
    """,
    unsafe_allow_html=True
)
