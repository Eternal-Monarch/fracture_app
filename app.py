import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# Mapping of model names to Google Drive file IDs
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "1ARBL_SK66Ppj7_kJ1Pe2FhH2olbTQHWY",
    "MobileNet (Keras)": "14YuV3qZb_6FI7pXoiJx69HxiDD4uNc_Q",
    "EfficientNetB3 (Keras)": "1cQA3_oH2XjDFK-ZE9D9YsP6Ya8fQiPOy"
}

# Function to download and load model
@st.cache_resource
def load_tensorflow_model(file_id, model_name):
    model_path = f"models/{model_name}.keras"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

# Preprocessing function
def preprocess_image_tf(uploaded_image, model):
    input_shape = model.input_shape[1:3]
    img = uploaded_image.resize(input_shape).convert("L")
    img_array = np.array(img) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit Page Configuration
st.set_page_config(page_title="Fracture Detection AI", page_icon="🩻", layout="centered")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3212/3212788.png", width=80)
    st.title("🩺 Medical AI Tool")
    theme = st.selectbox("App Theme", ["Light", "Dark"], index=0)
    selected_model_name = st.selectbox("Choose Model", options=list(model_ids.keys()))
    st.markdown("---")
    st.markdown("👨‍💻 **Developer:** Subhasish Dutta")

# Theme styles
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #0b0c10; color: #c5c6c7; }
            .stButton>button { background-color: #45a29e; color: white; }
            .stFileUploader>div { background-color: #1f2833; border: 1px solid #66fcf1; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body { background-color: #f0f2f6; color: #1f2833; }
            .stButton>button { background-color: #007acc; color: white; }
            .stFileUploader>div { background-color: #ffffff; border: 1px solid #cccccc; }
        </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #45a29e;'>🩻 AI-Powered Fracture Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a medical X-ray image to analyze potential fractures with advanced deep learning models.</p>", unsafe_allow_html=True)

# Load Model
file_id = model_ids[selected_model_name]
model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
st.success(f"✅ Loaded {selected_model_name}")

# File Upload Section
st.markdown("### 📤 Upload X-ray Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
        image_file = Image.open(uploaded_file).convert("RGB")
        processed_image = preprocess_image_tf(image_file, model)

        with st.spinner("🧪 Analyzing... Please wait"):
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_score = confidence if result == "Fracture Detected" else 1 - confidence

        st.markdown("---")
        st.subheader("📝 Diagnosis Report")
        if result == "Fracture Detected":
            st.error(f"⚠ **{result}** with confidence: **{confidence_score:.2f}**")
            st.info("📌 **Recommendation:** Please consult a certified medical professional for further examination.")
        else:
            st.success(f"✅ **{result}** with confidence: **{confidence_score:.2f}**")
            st.balloons()
            st.info("🎉 No fracture detected. Keep maintaining your bone health!")

    except Exception as e:
        st.error(f"❌ Error analyzing the image: {e}")
else:
    st.warning("ℹ Please upload an X-ray image to begin the analysis.")
