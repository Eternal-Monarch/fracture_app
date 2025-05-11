import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import time
from tensorflow.keras.models import load_model
from fpdf import FPDF
from datetime import datetime
import base64
from streamlit.components.v1 import html

# Model mappings for AI
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "10B53bzc1pYrQnBfDqBWrDpNmzWoOl9ac",
    "MobileNet (Keras)": "1mlfoy6kKXUwIciZW3nftmiMHOTzpy6_s",
    "EfficientNetB3 (Keras)": "1cQA3_oH2XjDFK-ZE9D9YsP6Ya8fQiPOy"
}

# Function to download and load models
@st.cache_resource
def load_tensorflow_model(file_id, model_name):
    model_path = f"models/{model_name}.keras"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

# Preprocessing function for image
def preprocess_image_tf(uploaded_image, model):
    input_shape = model.input_shape[1:3]
    img = uploaded_image.resize(input_shape).convert("L")
    img_array = np.array(img) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =============================================
# SIDEBAR COMPONENTS
# =============================================
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)
    st.markdown("<h1 style='font-size: 1.8rem; margin-bottom: 0.5rem;'>BoneScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0;'>Medical Imaging & Prescription System</p>", unsafe_allow_html=True)

    # Model selection dropdown
    selected_model_name = st.selectbox(
        "🧠 Select AI Model", 
        options=list(model_ids.keys()),
        help="Choose the deep learning model for analysis",
        index=0
    )
    
    # Model details expander (fixed HTML rendering issue)
    with st.expander("ℹ Model Details", expanded=False):
        if selected_model_name == "DenseNet169 (Keras)":
            st.markdown("""
                - *Type*: Dense Convolutional Network
                - *Depth*: 169 layers
                - *Strengths*: Feature reuse, parameter efficiency
                - *Best for*: High-resolution images with fine details
                - *Accuracy*: 94.2% (validation set)
            """)
        elif selected_model_name == "InceptionV3 (Keras)":
            st.markdown("""
                - *Type*: Inception architecture
                - *Depth*: 48 layers
                - *Strengths*: Multi-scale processing
                - *Best for*: General purpose medical imaging
                - *Accuracy*: 93.7% (validation set)
            """)
        elif selected_model_name == "MobileNet (Keras)":
            st.markdown("""
                - *Type*: Depthwise separable convolutions
                - *Depth*: 28 layers
                - *Strengths*: Fast inference, lightweight
                - *Best for*: Mobile/edge devices
                - *Accuracy*: 91.8% (validation set)
            """)
        elif selected_model_name == "EfficientNetB3 (Keras)":
            st.markdown("""
                - *Type*: Compound scaling
                - *Depth*: 48 layers
                - *Strengths*: State-of-the-art accuracy
                - *Best for*: Most accurate results
                - *Accuracy*: 95.1% (validation set)
            """)

# =============================================
# MAIN CONTENT
# =============================================
# Header with gradient
st.markdown("""
    <div class="header">
        <h1>BoneScan AI</h1>
        <h3>Advanced Fracture Detection System</h3>
    </div>
""", unsafe_allow_html=True)

# AI Model Information (fixed rendering)
st.markdown("""
    <h4>Available Models</h4>
    <ul>
        <li>DenseNet169</li>
        <li>InceptionV3</li>
        <li>MobileNet</li>
        <li>EfficientNetB3</li>
    </ul>

    <h4>Performance Metrics</h4>
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between;">
            <span>Accuracy</span>
            <span>92-96%</span>
        </div>
        <div style="height: 6px; background: var(--border); border-radius: 3px; margin: 0.3rem 0;">
            <div style="width: 94%; height: 100%; background: var(--primary); border-radius: 3px;"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# =============================================
# Main Content Section for Upload and Analysis
# =============================================

# Upload section for image
st.markdown("""
    <div class="card upload-card">
        <h2>📤 Upload X-ray Image</h2>
        <p>Supported formats: JPG, PNG (min. 512×512 pixels)</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop or click to browse files", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    key="file_uploader"
)

if uploaded_file:
    try:
        with st.spinner("Processing image..."):
            # Show loading animation
            loading_placeholder = st.empty()
            loading_placeholder.markdown("""<div style="text-align: center;">🔍 Processing...</div>""", unsafe_allow_html=True)
            time.sleep(1)  # Simulate processing for better UX

            image_file = Image.open(uploaded_file).convert("RGB")
            loading_placeholder.empty()
            
            # Display image preview
            st.markdown("### 🖼 X-ray Preview")
            st.image(image_file, caption="Uploaded X-ray", use_column_width=True, output_format="PNG", width=400)

            # Load model and analyze
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.markdown("🔄 Loading model...")
            progress_bar.progress(20)

            file_id = model_ids[selected_model_name]
            model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))

            status_text.markdown("🔍 Analyzing image...")
            progress_bar.progress(50)
            processed_image = preprocess_image_tf(image_file, model)

            prediction = model.predict(processed_image)
            confidence = prediction[0][0]
            result = "Fracture Detected" if confidence > 0.5 else "Normal"
            confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
            confidence_percent = confidence_score * 100

            # Results display
            st.markdown("### 📊 Analysis Results")
            st.markdown(f"**Diagnosis**: {result} with a confidence level of {confidence_percent:.1f}%")

            if result == "Fracture Detected":
                st.markdown(f"<div style='color: red;'>⚠ Urgent: Fracture Detected</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='color: green;'>✅ No Fracture Detected</div>", unsafe_allow_html=True)

            # Option to download the analysis report
            st.download_button(
                label="📄 Download Analysis Report",
                data=f"Diagnosis: {result}\nConfidence: {confidence_percent:.1f}%",
                file_name="bonescan_analysis_report.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Error analyzing the image: {str(e)}")
        st.markdown("<div class='emergency-alert'>Error processing the image. Please try again.</div>", unsafe_allow_html=True)

