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
st.set_page_config(page_title="Fracture Detection", page_icon="🪴", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🔍 AI Fracture Classification")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
    selected_model_name = st.selectbox("Select a Model", options=list(model_ids.keys()))
    st.info("*Developer:* Subhasish Dutta")

# Theme styles
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: #ffffff; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stFileUploader>div { background-color: #444444; border: 1px solid #555555; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            body { background-color: #ffffff; color: #000000; }
            .stButton>button { background-color: #008CBA; color: white; }
            .stFileUploader>div { background-color: #f5f5f5; border: 1px solid #dddddd; }
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Fracture Detection System</h1>", unsafe_allow_html=True)
st.write("📂 *Upload an X-ray image to detect fractures using AI.*")

# Load Model
file_id = model_ids[selected_model_name]
model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
st.success(f"✅ {selected_model_name} loaded successfully!")

# File Upload
uploaded_file = st.file_uploader("📂 Upload an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)
        image_file = Image.open(uploaded_file).convert("RGB")
        processed_image = preprocess_image_tf(image_file, model)

        with st.spinner("🔍 Analyzing the image..."):
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_score = confidence if result == "Fracture Detected" else 1 - confidence

        st.subheader("🔎 Results")
        if result == "Fracture Detected":
            st.error(f"⚠ *{result}* with confidence: *{confidence_score:.2f}*")
            st.write("🔗 *Recommendation:* Consult a healthcare professional.")
        else:
            st.success(f"✅ *{result}* with confidence: *{confidence_score:.2f}*")
            st.write("🎉 *No fracture detected! Maintain healthy bones.*")

    except Exception as e:
        st.error(f"🚨 Error processing the image: {e}")
else:
    st.info("💡 Please upload an image to start the analysis.")
