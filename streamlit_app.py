
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PotatoCare | AI Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f5f8f3;
}

.block-container {
    max-width: 1200px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.navbar {
    background: #153e2b;
    padding: 18px 28px;
    border-radius: 15px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 35px;
}

.brand {
    font-size: 23px;
    font-weight: 800;
}

.brand span {
    color: #a9df8e;
}

.nav-tag {
    color: #d8ead9;
    font-size: 13px;
}

.hero {
    background: linear-gradient(120deg, #173f2b, #286b43);
    padding: 45px;
    border-radius: 22px;
    color: white;
    margin-bottom: 30px;
}

.hero-label {
    color: #b9e7a5;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: 38px;
    color: white;
    margin: 12px 0;
}

.hero p {
    color: #e0eee2;
    font-size: 16px;
    max-width: 650px;
    line-height: 1.7;
}

.section-title {
    color: #173f2b;
    font-size: 23px;
    font-weight: 800;
    margin: 15px 0;
}

.panel {
    background: white;
    border: 1px solid #e3ebe2;
    border-radius: 18px;
    padding: 25px;
    min-height: 250px;
}

.panel h3 {
    color: #204c32;
    font-size: 19px;
}

.muted {
    color: #718174;
    font-size: 14px;
    line-height: 1.6;
}

.result-card {
    background: #eaf5e9;
    border: 1px solid #cde5ca;
    padding: 25px;
    border-radius: 16px;
    margin-top: 20px;
}

.result-label {
    color: #52745a;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.result-name {
    color: #17472d;
    font-size: 25px;
    font-weight: 800;
    margin: 10px 0;
}

.disease-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e3ebe2;
    min-height: 150px;
}

.disease-card h4 {
    color: #205638;
    margin-bottom: 10px;
}

.disease-card p {
    color: #718174;
    font-size: 13px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #879389;
    font-size: 12px;
    margin-top: 50px;
}

div.stButton > button {
    background: #24663e;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px 25px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #17472d;
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #f7faf6;
    border: 1px dashed #9db99e;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e3ebe2;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potato_model_v1.h5")


CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

DISPLAY_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy Leaf"
}


# ---------------- HEADER ----------------

st.markdown("""
<div class="navbar">
    <div class="brand">🌿 Potato<span>Care</span></div>
    <div class="nav-tag">AI-POWERED CROP HEALTH ANALYSIS</div>
</div>

<div class="hero">
    <div class="hero-label">SMART AGRICULTURE · DEEP LEARNING</div>
    <h1>Understand your crop's health.</h1>
    <p>
        Upload a potato leaf image and use our deep learning model
        to classify it as Early Blight, Late Blight, or Healthy.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- UPLOAD SECTION ----------------

st.markdown('<div class="section-title">Leaf Analysis</div>',
            unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <h3>📤 Upload your leaf image</h3>
        <p class="muted">
            Choose a clear JPG, JPEG, or PNG image of a potato leaf.
            For better results, use a well-lit image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded potato leaf",
                use_container_width=True
            )
        except Exception:
            st.error("Unable to read this image. Please upload another.")

with right:
    st.markdown("""
    <div class="panel">
        <h3>🔬 Analysis overview</h3>
        <p class="muted">
            Your uploaded image is resized and passed to the trained
            TensorFlow model for classification.
        </p>
        <hr>
        <p><b>Model:</b> Potato Disease Classifier</p>
        <p><b>Input:</b> 256 × 256 RGB</p>
        <p><b>Output:</b> 3 classes</p>
        <p><b>Framework:</b> TensorFlow / Keras</p>
    </div>
    """, unsafe_allow_html=True)

    analyse = st.button(
        "🔍 Analyse Leaf Image",
        disabled=uploaded_file is None
    )


# ---------------- PREDICTION ----------------

if uploaded_file is not None and analyse:

    try:
        with st.spinner("Analysing your leaf image..."):

            model = load_model()

            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((256, 256))

            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = tf.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array, verbose=0)

            predicted_index = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100

            predicted_class = CLASS_NAMES[predicted_index]
            display_name = DISPLAY_NAMES[predicted_class]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">PREDICTED CLASS</div>
            <div class="result-name">{display_name}</div>
            <p class="muted">
                This is the class selected by your trained
                image classification model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Model score", f"{confidence:.2f}%")
        st.progress(min(confidence / 100, 1.0))

        st.caption(
            "The model score is not a guarantee of diagnosis. "
            "Predictions depend on image quality and training data."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- DISEASE INFORMATION ----------------

st.markdown("---")
st.markdown(
    '<div class="section-title">What can PotatoCare identify?</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="disease-card">
        <h4>🍂 Early Blight</h4>
        <p>
            A fungal disease commonly associated with
            dark spots and ring-like patterns on leaves.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="disease-card">
        <h4>🌧️ Late Blight</h4>
        <p>
            A serious potato disease that can cause
            rapidly developing leaf lesions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="disease-card">
        <h4>🌱 Healthy Leaf</h4>
        <p>
            The model classifies the image as belonging
            to the healthy potato leaf category.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    PotatoCare · AI-Powered Potato Leaf Classification<br>
    Built with Python, Streamlit and TensorFlow
</div>
""", unsafe_allow_html=True)
