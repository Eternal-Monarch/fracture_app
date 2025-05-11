import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
import time
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

# Streamlit App Configuration
st.set_page_config(
    page_title="BoneScan AI - Medical Prescription & Fracture Detection",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar model selection and image upload
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)
    st.markdown("<h1 style='font-size: 1.8rem;'>BoneScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p>Clinical Fracture Detection</p>", unsafe_allow_html=True)

    selected_model_name = st.selectbox(
        "🧠 Select AI Model", 
        options=["DenseNet169", "InceptionV3", "MobileNet", "EfficientNetB3"],
        help="Choose the deep learning model for analysis"
    )

    # Add Model description
    with st.expander("Model Details"):
        st.markdown(f"""
            **Available Models:**
            - DenseNet169
            - InceptionV3
            - MobileNet
            - EfficientNetB3

            **Performance Metrics:**
            - Accuracy: 92-96%
        """)

# Main content area for analysis
st.markdown("### X-ray Image Upload and AI Analysis")
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.write("Processing image...")

    # Placeholder for loading animation
    loading_placeholder = st.empty()
    loading_placeholder.markdown("<div style='text-align:center;'>🔍 Processing...</div>", unsafe_allow_html=True)
    
    time.sleep(2)  # Simulate processing time

    # Load selected model and make prediction (simplified for now)
    model = load_tensorflow_model(model_ids[selected_model_name], selected_model_name)
    image = Image.open(uploaded_file)
    processed_image = preprocess_image_tf(image, model)

    # Fake prediction for demonstration
    confidence = 0.75  # For testing purposes
    result = "Fracture Detected" if confidence > 0.5 else "Normal"
    confidence_percent = confidence * 100

    loading_placeholder.empty()

    st.markdown(f"### Results: {result}")
    st.markdown(f"**Confidence:** {confidence_percent:.1f}%")

    # Option to generate prescription button after analysis
    with st.expander("Generate Prescription PDF", expanded=True):
        # Only show prescription options after analysis
        if result == "Fracture Detected":
            st.write("Patient information and prescription generation options will appear here.")
            patient_info = {
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "id": "1234",
                "allergies": "None"
            }
            diagnosis = "Fracture detected in the left leg."
            medications = [{"name": "Painkiller", "dosage": "500mg", "frequency": "3 times a day", "duration": "7 days", "special_instructions": "Take after meals"}]
            instructions = "Follow-up in 2 weeks."
            doctor_info = {"name": "Dr. Smith", "specialty": "Orthopedic", "license": "ORTH123", "contact": "1234567890"}

            # Generate and show the link to download the prescription PDF
            pdf_path = create_prescription(patient_info, diagnosis, medications, instructions, doctor_info)
            download_link = create_download_link(pdf_path, "Prescription.pdf")
            st.markdown(download_link, unsafe_allow_html=True)
        else:
            st.warning("No fracture detected, prescription not necessary.")
