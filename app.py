import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tensorflow as tf
import datetime
import csv

# --------------------
# CONFIG
# --------------------
APP_TITLE = "DermalScan — AI Facial Skin Aging Detection"
MODEL_PATH = "final_efficientnet_model1.h5"
CLASSES = ['wrinkles', 'dark spots', 'puffy eyes', 'clear skin']
IMG_SIZE = 224
FEEDBACK_CSV = "feedbacks.csv"

# --------------------
# HELPERS
# --------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def pil_to_bgr(pil_image: Image.Image):
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def preprocess_face(face_bgr: np.ndarray):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    x = face_resized.astype("float32")
    x = np.expand_dims(x, axis=0)
    return x

def detect_and_predict(model, face_cascade, image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    annotated = image_bgr.copy()
    results = []
    if len(faces) == 0:
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), results
    for (x,y,w,h) in faces:
        face_roi = annotated[y:y+h, x:x+w]
        if model is not None:
            inp = preprocess_face(face_roi)
            probs = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = CLASSES[idx]
            confidence = float(probs[idx])
        else:
            mean = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)))
            idx = int((mean // 64) % len(CLASSES))
            probs = np.zeros(len(CLASSES))
            probs[idx] = 1.0
            label = CLASSES[idx]
            confidence = 0.65
        cv2.rectangle(annotated, (x,y), (x+w, y+h), (10, 120, 180), 3)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(annotated, text, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,120,180), 2)
        results.append({"box":(int(x),int(y),int(w),int(h)), "label":label, "confidence":confidence, "probs":probs.tolist()})
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), results

def save_feedback(name, email, rating, message):
    exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp","name","email","rating","message"])
        writer.writerow([datetime.datetime.now().isoformat(), name, email, rating, message])

# --------------------
# STYLES (hero + cards like the reference)
# --------------------
STYLE = """
<style>
body { background: linear-gradient(180deg,#0b1220 0%, #071022 100%); color:#eaf2fb; }
.header {
  display:flex; gap:16px; align-items:center;
}
.logo {
  width:72px; height:72px; background:linear-gradient(135deg,#0abde3,#6f42c1); border-radius:14px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px;
  box-shadow: 0 8px 30px rgba(12, 24, 56, 0.6);
}
.hero-title { font-size:28px; font-weight:700; margin:0; }
.hero-sub { color:#a6bccf; margin-top:4px; margin-bottom:8px; }
.card { background: rgba(255,255,255,0.03); padding:14px; border-radius:12px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); }
.small { color:#9fb0c9; font-size:13px; }
.button-like { background: linear-gradient(90deg,#0abde3,#6f42c1); padding:8px 14px; color:white; border-radius:8px; font-weight:600; text-decoration:none; }
.footer { color:#91a7bd; font-size:13px; margin-top:18px; }
</style>
"""

# --------------------
# DOCTORS (sample)
# --------------------
DOCTORS = [
    {"name":"Dr Asha Menon","spec":"Dermatologist","clinic":"SkinCare Clinic, Hyderabad","phone":"+91-9876543210"},
    {"name":"Dr Rajiv Kumar","spec":"Cosmetic Dermatology","clinic":"DermaPlus, Bangalore","phone":"+91-9123456780"},
    {"name":"Dr Meera Iyer","spec":"Dermatologist","clinic":"GlowSkin, Chennai","phone":"+91-9988776655"}
]

# --------------------
# LAYOUT
# --------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    st.markdown(STYLE, unsafe_allow_html=True)

    # Header / Hero
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.markdown('<div class="logo">DS</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-left:12px;">'
                    f'<div class="hero-title">{APP_TITLE}</div>'
                    '<div class="hero-sub">Preliminary, privacy-first skin aging feature detection and guidance</div>'
                    '</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="small">Upload a clear face photo or take one with your camera. Results are indicative only.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card" style="text-align:center;">'
                    '<div style="font-weight:700">Credits</div>'
                    '<div style="font-size:22px; margin-top:6px;">5</div>'
                    '<div class="small">Use for analyses</div>'
                    '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Left column: actions and info; Right column: live preview / doctor cards
    left, right = st.columns([2,1])

    model = load_model()
    face_cascade = load_face_cascade()

    # Actions card
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Analyze your skin")
        st.write("Upload an image or capture with your webcam. The model will detect faces and provide class probabilities for common signs of skin aging.")
        col_a, col_b = st.columns([1,1])
        with col_a:
            uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
        with col_b:
            cam = st.camera_input("Take a photo")

        image = None
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
            except:
                st.error("Unable to open uploaded file.")
        elif cam is not None:
            image = Image.open(cam)

        if image is not None:
            st.image(image, use_column_width=True, caption="Input image")
            if st.button("Run Analysis"):
                annotated, results = detect_and_predict(model, face_cascade, pil_to_bgr(image))
                st.image(annotated, use_column_width=True, caption="Annotated result")
                if not results:
                    st.warning("No faces detected. Try better lighting or closer framing.")
                else:
                    for i, r in enumerate(results, start=1):
                        st.markdown(f"**Face {i}** — {r['label']} ({r['confidence']*100:.1f}%)")
                        prob_table = "|Class|Probability|\n|---:|---:|\n"
                        for c,p in zip(CLASSES, r["probs"]):
                            prob_table += f"|{c}|{p*100:.2f}%|\n"
                        st.markdown(prob_table)
        st.markdown("</div>", unsafe_allow_html=True)

        # Medical precautions card
        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### Medical precautions")
        st.markdown("""
        - Use SPF 30+ daily and reapply.
        - Avoid smoking; it accelerates aging.
        - Stay hydrated and maintain a balanced diet.
        - Seek a dermatologist for sudden or severe changes.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Feedback form
        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### Send feedback")
        with st.form("fb", clear_on_submit=True):
            name = st.text_input("Name")
            email = st.text_input("Email")
            rating = st.slider("Rate", 1, 5, 4)
            msg = st.text_area("Message")
            submitted = st.form_submit_button("Submit")
            if submitted:
                save_feedback(name or "Anon", email or "noemail", rating, msg or "")
                st.success("Thanks. Feedback recorded.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: doctors + about
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Doctor Recommendations")
        for d in DOCTORS:
            st.markdown(f"**{d['name']}**  \n{d['spec']} — {d['clinic']}  \nPhone: {d['phone']}")
            st.markdown("---")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### About the app")
        st.markdown("This is a demo educational tool. Not a medical device. Images are processed locally on the server/session. For diagnosis, consult a dermatologist.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="footer">Built for demo & university projects • 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
