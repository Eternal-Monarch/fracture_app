import streamlit as st
from fpdf import FPDF
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
import os
import gdown
from PIL import Image
import time

# Mapping of model names to Google Drive file IDs
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "10B53bzc1pYrQnBfDqBWrDpNmzWoOl9ac",
    "MobileNet (Keras)": "1mlfoy6kKXUwIciZW3nftmiMHOTzpy6_s",
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
    img = uploaded_image.resize(input_shape).convert("RGB")  # Convert to RGB using Pillow
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to create PDF Report
def create_pdf(patient_name, doctor_name, result, confidence_percent, selected_model_name, timestamp, logo_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set Font
    pdf.set_font("Arial", "B", 16)

    # Logo on the left side
    pdf.image(logo_path, 10, 8, 30)  # Ensure the correct path to your logo

    # Title Section
    pdf.cell(200, 10, "BoneScan AI - Fracture Detection Report", ln=True, align="C")
    pdf.ln(10)  # Line break

    # Report Info
    pdf.set_font("Arial", size=12)
    pdf.cell(100, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(100, 10, f"Doctor's Name: {doctor_name}", ln=True)
    pdf.cell(100, 10, f"Date: {timestamp.strftime('%Y-%m-%d')}", ln=True)
    pdf.cell(100, 10, f"Time: {timestamp.strftime('%H:%M:%S')}", ln=True)
    pdf.ln(10)  # Line break

    # Model and result
    pdf.cell(100, 10, f"Model Used: {selected_model_name}", ln=True)
    pdf.cell(100, 10, f"Result: {result}", ln=True)
    pdf.cell(100, 10, f"Confidence Level: {confidence_percent:.2f}%", ln=True)
    pdf.ln(10)  # Line break

    # Disclaimer
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 10, "Disclaimer: This tool provides an AI-assisted preliminary assessment and should not be used as a substitute for professional medical diagnosis. Always consult with a healthcare professional for confirmation of the results.")

    # Recommendations
    pdf.set_font("Arial", "B", 12)
    if result == "Fracture Detected":
        pdf.multi_cell(0, 10, "Recommendation: Immediate consultation with an orthopedic specialist is advised. Immobilize the affected area, apply ice, and seek urgent care.")
    else:
        pdf.multi_cell(0, 10, "Recommendation: No fracture detected. However, if pain persists, consult a healthcare provider for further evaluation.")

    # Save PDF to a file
    output_filename = "BoneScan_Report.pdf"
    pdf.output(output_filename)

    return output_filename

# Streamlit Page Configuration
st.set_page_config(
    page_title="BoneScan AI - Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for patient details
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)  # Replace with your logo path
    st.title("BoneScan AI")
    
    patient_name = st.text_input("Patient Name", "John Doe")
    doctor_name = st.text_input("Doctor's Name", "Dr. Smith")

    # Model selection
    selected_model_name = st.selectbox("Select AI Model", options=list(model_ids.keys()))
    st.markdown("### 📝 Report Instructions")
    st.markdown("""
    1. Upload an X-ray image
    2. Select the AI model for analysis
    3. View the analysis results
    4. Download the PDF report
    """)

# Main content
st.markdown("""
    <h1 style="text-align: center;">BoneScan AI - Fracture Detection</h1>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded X-ray", use_column_width=True)

    # Load the model and process the image
    with st.spinner("Processing image..."):
        file_id = model_ids[selected_model_name]
        model = load_tensorflow_model(file_id, selected_model_name)
        processed_image = preprocess_image_tf(image_file, model)
        prediction = model.predict(processed_image)
        confidence = prediction[0][0]

        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_percent = confidence * 100 if result == "Fracture Detected" else (1 - confidence) * 100

        timestamp = datetime.now()

        # Display results
        st.markdown(f"### Diagnosis Result: {result}")
        st.markdown(f"Confidence Level: {confidence_percent:.2f}%")

        # Generate PDF report
        if st.button("Download Report"):
            logo_path = "https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif"  # Change this to the actual logo path
            report_filename = create_pdf(patient_name, doctor_name, result, confidence_percent, selected_model_name, timestamp, logo_path)
            with open(report_filename, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name=report_filename,
                    mime="application/pdf"
                )
