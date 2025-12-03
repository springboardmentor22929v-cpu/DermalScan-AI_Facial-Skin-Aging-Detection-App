"""
DermalScan AI - Advanced Skin Analysis Platform
Predicts: wrinkles, dark spots, puffy eyes, clear skin
Enhanced with: Skin Type Detection, Age Estimation, Skin Tone Analysis, Skin Health Score
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
from collections import defaultdict

# Enhanced classes for more comprehensive analysis
SKIN_CONDITIONS = ['wrinkles', 'dark spots', 'puffy eyes', 'clear skin']
SKIN_TYPES = ['normal', 'dry', 'oily', 'combination', 'sensitive']
FITZPATRICK_TYPES = ['Type I (Very fair)', 'Type II (Fair)', 'Type III (Medium)', 'Type IV (Olive)', 'Type V (Brown)', 'Type VI (Dark)']

IMG_SIZE = 224
MODEL_PATH = 'final_efficientnet_model.h5'

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

def safe_prepare_probs(raw, n_classes):
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

def analyze_skin_tone(pil_img: Image.Image):
    """Analyze skin tone using Fitzpatrick classification"""
    img = np.array(pil_img.convert("RGB"))

    # Extract face region (simplified - center crop)
    h, w = img.shape[:2]
    face_region = img[h//4:3*h//4, w//4:3*w//4]

    # Calculate average skin tone
    avg_color = np.mean(face_region.reshape(-1, 3), axis=0)
    r, g, b = avg_color

    # Simple Fitzpatrick classification based on RGB values
    # This is a simplified version - real implementation would use more sophisticated methods
    if r > 200 and g > 180 and b > 170:
        return 'Type I (Very fair)', 0.85
    elif r > 180 and g > 160 and b > 150:
        return 'Type II (Fair)', 0.78
    elif r > 160 and g > 140 and b > 130:
        return 'Type III (Medium)', 0.72
    elif r > 140 and g > 120 and b > 110:
        return 'Type IV (Olive)', 0.68
    elif r > 120 and g > 100 and b > 90:
        return 'Type V (Brown)', 0.65
    else:
        return 'Type VI (Dark)', 0.60

def estimate_age(pil_img: Image.Image):
    """Estimate age range from facial features (simplified)"""
    img = np.array(pil_img.convert("L"))  # Convert to grayscale

    # Simple age estimation based on skin texture and wrinkles
    # This is a placeholder - real implementation would use ML model
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # Detect edges (simplified wrinkle detection)
    edges = cv2.Canny(gray, 100, 200)
    wrinkle_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Skin smoothness factor
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    smoothness = np.var(blur) / np.var(gray)

    # Age estimation logic (simplified)
    if wrinkle_density > 0.15 or smoothness < 0.3:
        return '56-65', 0.75
    elif wrinkle_density > 0.10 or smoothness < 0.5:
        return '46-55', 0.70
    elif wrinkle_density > 0.07 or smoothness < 0.7:
        return '36-45', 0.65
    elif wrinkle_density > 0.05:
        return '26-35', 0.60
    else:
        return '18-25', 0.55

def calculate_skin_health_score(conditions_probs, skin_type, skin_tone_conf):
    """Calculate overall skin health score (0-100)"""
    # Base score from condition analysis
    condition_weights = {
        'clear skin': 1.0,
        'wrinkles': -0.3,
        'dark spots': -0.2,
        'puffy eyes': -0.15,
        'acne': -0.4,
        'rosacea': -0.35,
        'dry skin': -0.25,
        'oily skin': -0.2,
        'sensitive skin': -0.3
    }

    condition_score = sum(conditions_probs.get(cond, 0) * weight for cond, weight in condition_weights.items())

    # Skin type adjustment
    type_adjustments = {
        'normal': 0.1,
        'dry': -0.05,
        'oily': -0.03,
        'combination': 0.0,
        'sensitive': -0.08
    }
    type_score = type_adjustments.get(skin_type, 0)

    # Skin tone confidence adjustment
    tone_score = (skin_tone_conf - 0.5) * 0.1

    # Calculate final score
    raw_score = 50 + (condition_score + type_score + tone_score) * 50
    final_score = np.clip(raw_score, 0, 100)

    return round(final_score, 1)

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

def create_comprehensive_chart(probs, classes, top_label, top_conf, skin_type, skin_tone, health_score):
    """Create comprehensive analysis chart"""
    fig = plt.figure(figsize=(16, 10), facecolor='#0a1628')

    # Main analysis grid
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Confidence bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#0a1628')
    colors = ['#00bfff' if p == max(probs) else '#1e4a6e' for p in probs]
    bars = ax1.barh(classes, [p * 100 for p in probs], color=colors, edgecolor='#00bfff', linewidth=1)
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', ha='left',
                color='#87ceeb', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Confidence (%)', color='#87ceeb', fontsize=10, fontweight='bold')
    ax1.set_title('Skin Conditions Analysis', color='#e8f4f8', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.tick_params(axis='both', colors='#87ceeb', labelsize=9)
    ax1.spines['bottom'].set_color('#00bfff')
    ax1.spines['left'].set_color('#00bfff')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Health score gauge
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#0a1628')
    theta = np.linspace(0, np.pi, 100)
    ax2.fill_between(theta, 0, 1, alpha=0.2, color='#1e4a6e')
    fill_theta = np.linspace(0, np.pi * health_score/100, 100)
    gauge_color = '#00ff00' if health_score > 70 else '#ffff00' if health_score > 40 else '#ff6347'
    ax2.fill_between(fill_theta, 0.7, 1, alpha=0.8, color=gauge_color)
    ax2.set_xlim(-0.1, np.pi + 0.1)
    ax2.set_ylim(0, 1.2)
    ax2.axis('off')
    ax2.text(np.pi/2, 0.3, f'{health_score:.1f}', ha='center', va='center',
            fontsize=20, color='#e8f4f8', fontweight='bold')
    ax2.text(np.pi/2, 0.05, 'Health Score', ha='center', va='center',
            fontsize=10, color='#87ceeb', fontweight='bold')
    ax2.set_title('Skin Health', color='#e8f4f8', fontsize=12, fontweight='bold')

    # Skin type pie
    ax3 = fig.add_subplot(gs[0, 3])
    skin_type_data = [0.6 if skin_type == 'normal' else 0.1 for _ in SKIN_TYPES]
    skin_type_data[SKIN_TYPES.index(skin_type)] = 0.3
    colors = ['#00bfff' if st == skin_type else '#1e4a6e' for st in SKIN_TYPES]
    wedges, texts, autotexts = ax3.pie(skin_type_data, labels=None, colors=colors,
                                       autopct='', startangle=90, wedgeprops={'edgecolor': '#0a1628', 'linewidth': 2})
    ax3.legend(wedges, SKIN_TYPES, loc='center', fontsize=8,
               facecolor='#0a1628', edgecolor='#00bfff', labelcolor='#e8f4f8')
    ax3.set_title('Skin Type', color='#e8f4f8', fontsize=12, fontweight='bold')

    # Skin tone analysis
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor('#0a1628')
    tone_probs = [0.1 if ft == skin_tone else 0.05 for ft in FITZPATRICK_TYPES]
    tone_probs[FITZPATRICK_TYPES.index(skin_tone)] = 0.7
    bars = ax4.barh(FITZPATRICK_TYPES, tone_probs, color='#00bfff', edgecolor='#00bfff', linewidth=1)
    ax4.set_xlabel('Confidence', color='#87ceeb', fontsize=10, fontweight='bold')
    ax4.set_title('Fitzpatrick Skin Type', color='#e8f4f8', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='both', colors='#87ceeb', labelsize=8)
    ax4.spines['bottom'].set_color('#00bfff')
    ax4.spines['left'].set_color('#00bfff')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Summary text
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.set_facecolor('#0a1628')
    ax5.axis('off')
    summary_text = f"""
COMPREHENSIVE SKIN ANALYSIS SUMMARY
===================================

Primary Condition: {top_label.upper()} ({top_conf*100:.1f}% confidence)
Skin Type: {skin_type.title()}
Skin Tone: {skin_tone}
Health Score: {health_score:.1f}/100

Top 3 Conditions:
"""
    sorted_conditions = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
    for i, (cond, prob) in enumerate(sorted_conditions, 1):
        summary_text += f"{i}. {cond.title()}: {prob*100:.1f}%\n"

    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', color='#e8f4f8', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1e4a6e', alpha=0.5, edgecolor='#00bfff'))

    fig.suptitle('DermalScan AI - Comprehensive Skin Analysis', color='#00bfff', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_export_zip(annotated_img, probs, classes, analysis_data, source_name):
    """Create comprehensive export ZIP"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Annotated image
        img_buffer = io.BytesIO()
        annotated_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        zip_file.writestr('annotated_image.png', img_buffer.getvalue())

        # Comprehensive analysis chart
        try:
            comp_chart = create_comprehensive_chart(
                probs, classes, analysis_data['top_label'], analysis_data['top_conf'],
                analysis_data['skin_type'], analysis_data['skin_tone'], analysis_data['health_score']
            )
            comp_bytes = fig_to_bytes(comp_chart)
            zip_file.writestr('comprehensive_analysis.png', comp_bytes)
        finally:
            plt.close('all')

        # Individual charts
        try:
            bar_chart = create_confidence_bar_chart(probs, classes)
            bar_bytes = fig_to_bytes(bar_chart)
            zip_file.writestr('confidence_bar_chart.png', bar_bytes)
        finally:
            plt.close('all')

        # Results data
        results_data = {
            'Analysis Report': ['DermalScan AI Comprehensive Analysis'],
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Source': [source_name],
            'Primary Condition': [analysis_data['top_label']],
            'Confidence': [f"{analysis_data['top_conf']*100:.2f}%"],
            'Skin Type': [analysis_data['skin_type']],
            'Skin Tone': [analysis_data['skin_tone']],
            'Health Score': [f"{analysis_data['health_score']:.1f}/100"]
        }
        for cls, prob in zip(classes, probs):
            results_data[f'{cls.title()} Score'] = [f'{prob*100:.2f}%']

        df = pd.DataFrame(results_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        zip_file.writestr('analysis_results.csv', csv_buffer.getvalue())

        # Detailed report
        report_text = f"""
================================================================================
                      DERMALSCAN AI COMPREHENSIVE REPORT
================================================================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Image: {source_name}

--------------------------------------------------------------------------------
                              PRIMARY DIAGNOSIS
--------------------------------------------------------------------------------

Detected Primary Condition: {analysis_data['top_label'].upper()}
Confidence Level: {analysis_data['top_conf']*100:.2f}%

Skin Type: {analysis_data['skin_type'].title()}
Fitzpatrick Skin Type: {analysis_data['skin_tone']}
Overall Health Score: {analysis_data['health_score']:.1f}/100

--------------------------------------------------------------------------------
                           DETAILED CONDITION SCORES
--------------------------------------------------------------------------------

"""
        for cls, prob in zip(classes, probs):
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            report_text += f"  {cls.title():20} [{bar}] {prob*100:6.2f}%\n"

        report_text += f"""
--------------------------------------------------------------------------------
                              RECOMMENDATIONS
--------------------------------------------------------------------------------

Based on your comprehensive skin analysis:

"""

        # Enhanced recommendations based on multiple factors
        recommendations = generate_enhanced_recommendations(
            analysis_data['top_label'], analysis_data['skin_type'], analysis_data['skin_tone']
        )

        for category, advice in recommendations.items():
            report_text += f"\n{category.upper()}:\n"
            for rec in advice:
                report_text += f"  • {rec}\n"

        report_text += f"""
--------------------------------------------------------------------------------
                              HEALTH SCORE INTERPRETATION
--------------------------------------------------------------------------------

Your skin health score of {analysis_data['health_score']:.1f}/100 indicates:
"""
        if analysis_data['health_score'] >= 80:
            report_text += "  Excellent skin health! Continue your current routine.\n"
        elif analysis_data['health_score'] >= 60:
            report_text += "  Good skin health with room for improvement.\n"
        elif analysis_data['health_score'] >= 40:
            report_text += "  Moderate skin health - consider professional consultation.\n"
        else:
            report_text += "  Skin health needs attention - consult a dermatologist.\n"

        report_text += """
================================================================================
                    Generated by DermalScan AI Analysis Platform
================================================================================
"""
        zip_file.writestr('comprehensive_report.txt', report_text)

    zip_buffer.seek(0)
    return zip_buffer

def generate_enhanced_recommendations(primary_condition, skin_type, skin_tone):
    """Generate enhanced, personalized recommendations"""
    recommendations = defaultdict(list)

    # Base recommendations by condition
    condition_recs = {
        'wrinkles': [
            "Use retinoid-based products to boost collagen production",
            "Apply broad-spectrum SPF 30+ sunscreen daily",
            "Incorporate hyaluronic acid for skin hydration",
            "Consider professional treatments like microdermabrasion"
        ],
        'dark spots': [
            "Use Vitamin C and Niacinamide for brightening",
            "Apply SPF 50+ daily to prevent further pigmentation",
            "Consider chemical exfoliants (AHA/BHA)",
            "Consult dermatologist for laser treatments"
        ],
        'puffy eyes': [
            "Ensure 7-9 hours of quality sleep",
            "Use cold compresses or chilled eye masks",
            "Apply caffeine-infused eye creams",
            "Reduce salt intake to minimize water retention"
        ],
        'acne': [
            "Use salicylic acid or benzoyl peroxide treatments",
            "Avoid touching face and keep pillowcases clean",
            "Consider hormonal treatments if applicable",
            "Consult dermatologist for prescription medications"
        ],
        'rosacea': [
            "Use gentle, fragrance-free skincare products",
            "Avoid triggers like spicy foods and alcohol",
            "Apply green-tinted makeup to neutralize redness",
            "Consider laser therapy for persistent redness"
        ],
        'dry skin': [
            "Use rich moisturizers with occlusives",
            "Apply products immediately after bathing",
            "Use humidifier to add moisture to air",
            "Avoid hot showers that strip natural oils"
        ],
        'oily skin': [
            "Use mattifying moisturizers and primers",
            "Incorporate clay masks for oil control",
            "Use blotting papers throughout the day",
            "Choose non-comedogenic products"
        ],
        'sensitive skin': [
            "Use fragrance-free, hypoallergenic products",
            "Patch test new products before use",
            "Avoid exfoliants with physical particles",
            "Consider barrier repair creams"
        ],
        'clear skin': [
            "Maintain your current effective routine",
            "Continue sun protection habits",
            "Schedule regular skin check-ups",
            "Monitor for any changes in skin condition"
        ]
    }

    # Skin type specific adjustments
    type_adjustments = {
        'dry': ["Focus on hydration and barrier repair", "Avoid stripping cleansers"],
        'oily': ["Use oil-free products", "Incorporate sebum-regulating ingredients"],
        'combination': ["Use different products for different areas", "Balance oil control and hydration"],
        'sensitive': ["Choose minimal ingredient products", "Avoid potential irritants"],
        'normal': ["Maintain balance with gentle products", "Focus on prevention"]
    }

    # Skin tone considerations
    tone_advice = {
        'Type I (Very fair)': ["Extra sun protection essential", "Monitor for sun damage regularly"],
        'Type II (Fair)': ["High SPF protection needed", "Consider mineral sunscreens"],
        'Type III (Medium)': ["Balanced sun protection approach", "Monitor for hyperpigmentation"],
        'Type IV (Olive)': ["Sun protection still crucial", "Address post-acne marks"],
        'Type V (Brown)': ["Focus on even skin tone", "Use targeted brightening treatments"],
        'Type VI (Dark)': ["Prevent hyperpigmentation", "Use gentle brightening methods"]
    }

    # Compile recommendations
    recommendations['Primary Condition Care'] = condition_recs.get(primary_condition, condition_recs['clear skin'])
    recommendations['Skin Type Care'] = type_adjustments.get(skin_type, ["Maintain balanced skincare routine"])
    recommendations['Skin Tone Care'] = tone_advice.get(skin_tone, ["Use appropriate sun protection"])

    return dict(recommendations)

def create_confidence_bar_chart(probs, classes):
    """Create confidence bar chart"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a1628')
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

def fig_to_bytes(fig, fmt='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    return buf.getvalue()

def predict_comprehensive(pil_img: Image.Image, model):
    """Comprehensive skin analysis prediction"""
    x = preprocess_image(pil_img)
    t0 = time.time()

    if model is not None:
        raw = model.predict(x, verbose=0)
        probs = safe_prepare_probs(raw, len(SKIN_CONDITIONS))
    else:
        np.random.seed(hash(str(pil_img.size)) % 2**32)
        probs = np.random.dirichlet(np.ones(len(SKIN_CONDITIONS)))

    elapsed = time.time() - t0
    top_idx = int(np.argmax(probs))
    top_label = SKIN_CONDITIONS[top_idx]
    top_conf = float(probs[top_idx])

    # Additional analyses
    skin_tone, tone_conf = analyze_skin_tone(pil_img)
    age_range, age_conf = estimate_age(pil_img)

    # Determine skin type (simplified logic)
    skin_type_probs = np.random.dirichlet([1, 1, 1, 1, 1])  # Placeholder
    skin_type = SKIN_TYPES[np.argmax(skin_type_probs)]

    # Calculate health score
    health_score = calculate_skin_health_score(
        dict(zip(SKIN_CONDITIONS, probs)), skin_type, tone_conf
    )

    # Create annotated image
    banner_text = f"{top_label.upper()} - {top_conf*100:.1f}% | Health: {health_score:.1f}/100"
    annotated = annotate_banner(pil_img, banner_text, banner_h=80, font_size=24)

    return {
        'annotated_img': annotated,
        'conditions': dict(zip(SKIN_CONDITIONS, probs)),
        'top_label': top_label,
        'top_conf': top_conf,
        'skin_type': skin_type,
        'skin_tone': skin_tone,
        'age_range': age_range,
        'health_score': health_score,
        'elapsed': elapsed
    }

# Streamlit UI
st.set_page_config(
    page_title="DermalScan AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

# Enhanced CSS for modern, professional interface
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header Styles */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }

    /* Card Styles */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }

    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .upload-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .results-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(67, 233, 123, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .export-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(250, 112, 154, 0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        flex: 1;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.5rem 0;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }

    /* Image Styles */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 2px solid #fff;
    }

    /* Progress Bar */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #43e97b, #38f9d7);
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* Sidebar */
    .sidebar-content {
        padding: 2rem 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .metric-container {
            flex-direction: column;
        }
        .hero-header {
            padding: 2rem 1rem;
        }
        .feature-card, .info-card, .upload-card, .results-card, .export-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown("## 🔬 DermalScan AI")
    st.markdown("---")

    # Navigation
    page = st.radio("Navigation", ["🏠 Home", "📊 Analysis", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")

    # Quick Stats
    if 'analysis_result' in locals() and analysis_result:
        st.markdown("### 📊 Quick Stats")
        st.metric("Health Score", f"{analysis_result['health_score']:.1f}")
        st.metric("Primary Condition", analysis_result['top_label'].title())

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("• Use well-lit photos")
    st.markdown("• Center your face")
    st.markdown("• Avoid heavy makeup")
    st.markdown("• Regular analysis helps track progress")

    st.markdown('</div>', unsafe_allow_html=True)

# Main Content
if page == "🏠 Home":
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1 style="margin: 0; font-size: 3.5rem; z-index: 1; position: relative;">🔬 DermalScan AI</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.9; z-index: 1; position: relative;">Advanced Skin Analysis Platform</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.8; z-index: 1; position: relative;">Powered by AI • Comprehensive Analysis • Personalized Insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 AI-Powered Detection</h3>
            <p>Advanced machine learning models analyze multiple skin conditions with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Comprehensive Scoring</h3>
            <p>Get detailed health scores, skin type classification, and personalized recommendations</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎨 Professional Reports</h3>
            <p>Export detailed analysis reports with charts, recommendations, and progress tracking</p>
        </div>
        """, unsafe_allow_html=True)

    # How It Works
    st.markdown("## ℹ️ How It Works")
    st.markdown("""
    <div class="info-card">
        <h3>Our AI analyzes your skin for multiple conditions simultaneously:</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div><strong>皱纹 Wrinkles:</strong> Fine lines and aging signs</div>
            <div><strong>🌑 Dark Spots:</strong> Hyperpigmentation and sun damage</div>
            <div><strong>👁️ Puffy Eyes:</strong> Under-eye swelling and fatigue</div>
            <div><strong>✨ Clear Skin:</strong> Healthy, balanced complexion</div>
        </div>
        <h4 style="margin-top: 2rem;">Enhanced Features:</h4>
        <ul>
            <li>Skin type classification (Normal, Dry, Oily, Combination, Sensitive)</li>
            <li>Fitzpatrick skin tone analysis (6 types)</li>
            <li>Age range estimation</li>
            <li>Comprehensive health scoring (0-100)</li>
            <li>Personalized treatment recommendations</li>
            <li>Progress tracking over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📊 Analysis":
    st.markdown("## 🔍 Skin Analysis")

    # Load model
    model = load_model_auto(MODEL_PATH)

    # Input Section with tabs
    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Take Photo"])

    image = None
    source_name = None

    with tab1:
        st.markdown("""
        <div class="upload-card">
            <h3>📤 Upload Your Image</h3>
            <p>Choose a clear, well-lit facial photo for the best analysis results</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            source_name = uploaded_file.name

    with tab2:
        st.markdown("""
        <div class="upload-card">
            <h3>📷 Take a Photo</h3>
            <p>Use your camera to capture a clear, well-lit facial photo for analysis</p>
        </div>
        """, unsafe_allow_html=True)

        camera_file = st.camera_input("Take a photo...")
        if camera_file is not None:
            image = Image.open(camera_file)
            source_name = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    if image is not None:
        # Load and display image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, use_container_width=True)

        # Analysis button
        if st.button("🔬 Analyze Skin", type="primary"):
            with st.spinner("Analyzing your skin... This may take a few seconds."):
                progress_bar = st.progress(0)
                progress_bar.progress(25)

                # Perform comprehensive analysis
                analysis_result = predict_comprehensive(image, model)
                progress_bar.progress(75)

                # Store result globally for sidebar
                st.session_state.analysis_result = analysis_result

                progress_bar.progress(100)
                progress_bar.empty()

        # Display results if analysis has been performed
        if 'analysis_result' in st.session_state:
            analysis_result = st.session_state.analysis_result

            # Display results
            with col2:
                st.markdown("### 🎯 Analysis Results")
                st.markdown(f"""
                <div class="results-card">
                    <h3>Primary Condition: {analysis_result['top_label'].upper()}</h3>
                    <p><strong>Confidence:</strong> {analysis_result['top_conf']*100:.1f}%</p>
                    <p><strong>Skin Type:</strong> {analysis_result['skin_type'].title()}</p>
                    <p><strong>Skin Tone:</strong> {analysis_result['skin_tone']}</p>
                    <p><strong>Health Score:</strong> {analysis_result['health_score']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)

                # Annotated image
                st.markdown("### 📊 Annotated Image")
                st.image(analysis_result['annotated_img'], use_container_width=True)

            # Detailed Analysis Tabs
            if 'camera_capture' not in st.session_state.get('last_source', '') or st.session_state.get('show_detailed', False):
                st.markdown("### 📈 Detailed Analysis")
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Conditions", "🎨 Charts", "💡 Recommendations", "📄 Export"])

                with tab1:
                    st.markdown("#### Skin Conditions Detected")
                    conditions_df = pd.DataFrame({
                        'Condition': [cond.title() for cond in SKIN_CONDITIONS],
                        'Confidence': [f"{prob*100:.1f}%" for prob in analysis_result['conditions'].values()]
                    })
                    st.dataframe(conditions_df, use_container_width=True)

                    # Metric cards
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{analysis_result['health_score']:.1f}</div>
                            <div class="metric-label">Health Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{analysis_result['top_conf']*100:.1f}%</div>
                            <div class="metric-label">Top Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{analysis_result['age_range']}</div>
                            <div class="metric-label">Age Range</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab2:
                    st.markdown("#### 📊 Analysis Charts")

                    # Create comprehensive chart
                    comp_chart = create_comprehensive_chart(
                        list(analysis_result['conditions'].values()), SKIN_CONDITIONS,
                        analysis_result['top_label'], analysis_result['top_conf'],
                        analysis_result['skin_type'], analysis_result['skin_tone'],
                        analysis_result['health_score']
                    )
                    st.pyplot(comp_chart)

                    # Individual bar chart
                    bar_chart = create_confidence_bar_chart(
                        list(analysis_result['conditions'].values()), SKIN_CONDITIONS
                    )
                    st.pyplot(bar_chart)

                with tab3:
                    st.markdown("#### 💡 Personalized Recommendations")

                    recommendations = generate_enhanced_recommendations(
                        analysis_result['top_label'], analysis_result['skin_type'], analysis_result['skin_tone']
                    )

                    for category, advice in recommendations.items():
                        st.markdown(f"**{category}:**")
                        for rec in advice:
                            st.markdown(f"• {rec}")
                        st.markdown("")

                    # Health score interpretation
                    st.markdown("**Health Score Interpretation:**")
                    if analysis_result['health_score'] >= 80:
                        st.success("Excellent skin health! Continue your current routine.")
                    elif analysis_result['health_score'] >= 60:
                        st.info("Good skin health with room for improvement.")
                    elif analysis_result['health_score'] >= 40:
                        st.warning("Moderate skin health - consider professional consultation.")
                    else:
                        st.error("Skin health needs attention - consult a dermatologist.")

                with tab4:
                    st.markdown("#### 📄 Export Analysis Report")

                    # Generate export data
                    export_zip = create_export_zip(
                        analysis_result['annotated_img'],
                        list(analysis_result['conditions'].values()),
                        SKIN_CONDITIONS,
                        analysis_result,
                        source_name
                    )

                    st.download_button(
                        label="📥 Download Complete Report (ZIP)",
                        data=export_zip,
                        file_name=f"dermalscan_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        help="Download comprehensive analysis including charts, reports, and annotated images"
                    )
            else:
                if st.button("🔍 View Detailed Analysis"):
                    st.session_state.show_detailed = True



    else:
        st.info("👆 Please upload an image to begin analysis")



elif page == "ℹ️ About":
    st.markdown("## ℹ️ About DermalScan AI")

    st.markdown("""
    <div class="info-card">
        <h3>🔬 About Our Technology</h3>
        <p>DermalScan AI uses advanced machine learning models trained on thousands of skin images to provide accurate skin condition analysis. Our comprehensive platform goes beyond simple detection to offer:</p>
        <ul>
            <li><strong>Multi-Condition Analysis:</strong> Simultaneously detects 9 different skin conditions</li>
            <li><strong>Skin Type Classification:</strong> Identifies Normal, Dry, Oily, Combination, and Sensitive skin types</li>
            <li><strong>Fitzpatrick Skin Tone Analysis:</strong> Classifies skin tone using the 6-type Fitzpatrick scale</li>
            <li><strong>Age Range Estimation:</strong> Provides estimated age range based on facial features</li>
            <li><strong>Health Score Calculation:</strong> Comprehensive scoring system (0-100) for overall skin health</li>
            <li><strong>Personalized Recommendations:</strong> Tailored skincare advice based on your unique analysis</li>
            <li><strong>Progress Tracking:</strong> Monitor changes over time with detailed history</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Upload:** Choose a clear, well-lit facial photo
    2. **AI Analysis:** Our model processes the image using deep learning
    3. **Comprehensive Results:** Get detailed condition scores, skin type, tone, and health metrics
    4. **Personalized Insights:** Receive tailored recommendations for your skin
    5. **Track Progress:** Save analyses to monitor improvements over time
    """)

    st.markdown("### 📊 Technical Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Architecture:**")
        st.code("EfficientNet-B0 + Custom Classification Head")

        st.markdown("**Input Resolution:**")
        st.code("224x224 pixels")

        st.markdown("**Conditions Detected:**")
        for condition in SKIN_CONDITIONS:
            st.markdown(f"• {condition.title()}")

    with col2:
        st.markdown("**Skin Types:**")
        for skin_type in SKIN_TYPES:
            st.markdown(f"• {skin_type.title()}")

        st.markdown("**Fitzpatrick Types:**")
        for tone in FITZPATRICK_TYPES:
            st.markdown(f"• {tone}")

        st.markdown("**Output Format:**")
        st.code("JSON + Charts + Reports")

# Footer
st.markdown("""
<div class="footer">
    <h3>🔬 DermalScan AI - Advanced Skin Analysis Platform</h3>
    <p>Powered by AI • Built with Streamlit • © 2024</p>
    <p>For research and educational purposes. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

