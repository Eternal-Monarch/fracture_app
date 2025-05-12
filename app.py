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

# Model mappings for fracture detection
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "1ARBL_SK66Ppj7_kJ1Pe2FhH2olbTQHWY",
    "MobileNet (Keras)": "14YuV3qZb_6FI7pXoiJx69HxiDD4uNc_Q",
    "EfficientNetB3 (Keras)": "1cQA3_oH2XjDFK-ZE9D9YsP6Ya8fQiPOy"
}

# Function to download and load fracture detection model
@st.cache_resource
def load_tensorflow_model(file_id, model_name):
    model_path = f"models/{model_name}.keras"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

# Preprocessing function for fracture detection
def preprocess_image_tf(uploaded_image, model):
    input_shape = model.input_shape[1:3]
    img = uploaded_image.resize(input_shape).convert("L")
    img_array = np.array(img) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# PDF Prescription Generator Class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'MEDICAL PRESCRIPTION', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_prescription(patient_info, diagnosis, medications, instructions, doctor_info, fracture_image, confidence_percentage, model_name):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header with clinic info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 5, "BoneScan AI Medical Center", 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, "123 Medical Drive, Healthcare City", 0, 1, 'C')
    pdf.cell(0, 5, "Phone: (123) 456-7890 | License: MED123456", 0, 1, 'C')
    pdf.ln(10)
    
    # Date and prescription ID
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 0, 1, 'R')
    pdf.cell(0, 5, f"Prescription ID: RX-{datetime.now().strftime('%Y%m%d%H%M')}", 0, 1, 'R')
    pdf.ln(5)
    
    # Patient information box
    pdf.set_fill_color(240, 240, 240)
    pdf.rect(10, 45, 190, 30, 'F')
    pdf.set_font('Arial', 'B', 12)
    pdf.set_xy(15, 50)
    pdf.cell(0, 5, "PATIENT INFORMATION", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_xy(15, 57)
    pdf.cell(40, 5, f"Name: {patient_info['name']}", 0, 0)
    pdf.cell(40, 5, f"Age: {patient_info['age']}", 0, 0)
    pdf.cell(40, 5, f"Gender: {patient_info['gender']}", 0, 1)
    pdf.set_xy(15, 64)
    pdf.cell(40, 5, f"Patient ID: {patient_info['id']}", 0, 0)
    pdf.cell(40, 5, f"Allergies: {patient_info['allergies']}", 0, 1)
    pdf.ln(10)
    
    # Diagnosis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "DIAGNOSIS", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, diagnosis)
    pdf.ln(10)
    
    # Medications
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "PRESCRIBED MEDICATIONS", 0, 1)
    pdf.set_font('Arial', '', 11)
    
    # Table header
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(60, 8, "Medication", 1, 0, 'C', 1)
    pdf.cell(30, 8, "Dosage", 1, 0, 'C', 1)
    pdf.cell(30, 8, "Frequency", 1, 0, 'C', 1)
    pdf.cell(30, 8, "Duration", 1, 0, 'C', 1)
    pdf.cell(40, 8, "Instructions", 1, 1, 'C', 1)
    
    # Medication rows
    pdf.set_fill_color(255, 255, 255)
    for med in medications:
        pdf.cell(60, 8, med['name'], 1)
        pdf.cell(30, 8, med['dosage'], 1)
        pdf.cell(30, 8, med['frequency'], 1)
        pdf.cell(30, 8, med['duration'], 1)
        pdf.cell(40, 8, med['special_instructions'], 1, 1)
    pdf.ln(10)
    
    # Fracture Image and Confidence
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "FRACTURE IMAGE & CONFIDENCE", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"Model: {model_name}", 0, 1)
    pdf.cell(0, 10, f"Confidence of Fracture: {confidence_percentage:.2f}%", 0, 1)
    
    if fracture_image:
        fracture_image_path = f"fracture_image.jpg"
        fracture_image.save(fracture_image_path)
        pdf.image(fracture_image_path, x=10, w=190)
    pdf.ln(10)
    
    # Additional Instructions
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "ADDITIONAL INSTRUCTIONS", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, instructions)
    pdf.ln(15)
    
    # Doctor information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "PRESCRIBING PHYSICIAN", 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f"Name: Dr. {doctor_info['name']}", 0, 1)
    pdf.cell(0, 7, f"Specialty: {doctor_info['specialty']}", 0, 1)
    pdf.cell(0, 7, f"License: {doctor_info['license']}", 0, 1)
    pdf.cell(0, 7, f"Contact: {doctor_info['contact']}", 0, 1)
    pdf.ln(10)
    
    # Signature line
    pdf.line(120, pdf.get_y(), 180, pdf.get_y())
    pdf.set_xy(120, pdf.get_y() + 2)
    pdf.cell(60, 5, "Doctor's Signature", 0, 0, 'C')
    
    # Disclaimer
    pdf.set_font('Arial', 'I', 10)
    pdf.ln(10)
    pdf.multi_cell(0, 7, "AI Model Disclaimer: This report is generated by an AI model for research purposes only. Always consult with a qualified healthcare provider for medical diagnosis and treatment.")
    
    # Save PDF
    pdf_path = "medical_prescription_with_image.pdf"
    pdf.output(pdf_path)
    return pdf_path

def create_download_link(pdf_path, filename):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Streamlit App Configuration
st.set_page_config(page_title="BoneScan AI - Fracture Detection & Prescription", page_icon="🦴", layout="wide", initial_sidebar_state="expanded")

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=100)
    st.title("BoneScan AI")
    
    # Navigation buttons
    st.markdown("### Navigation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🦴 Fracture Detection"):
            st.session_state.current_page = 'fracture_detection'
    with col2:
        if st.button("💊 Prescription"):
            st.session_state.current_page = 'prescription'

# Fracture Detection Page
def show_fracture_detection():
    selected_model_name = st.selectbox("🧠 Select AI Model", options=list(model_ids.keys()), help="Choose the deep learning model for analysis")
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_file = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True, output_format="PNG")
        
        # Load selected model
        file_id = model_ids[selected_model_name]
        model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
        
        processed_image = preprocess_image_tf(image_file, model)
        prediction = model.predict(processed_image)
        confidence = prediction[0][0]
        
        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
        confidence_percent = confidence_score * 100
        
        # Save fracture image and confidence in session state for prescription page
        st.session_state.fracture_image = uploaded_file
        st.session_state.confidence_percentage = confidence_percent
        st.session_state.selected_model_name = selected_model_name
        
        # Results display
        st.markdown(f"Result: {result}")
        st.markdown(f"Confidence: {confidence_percent:.2f}%")

# Prescription Generator Page
def show_prescription_generator():
    with st.form("prescription_form"):
        patient_name = st.text_input("Full Name*")
        patient_age = st.text_input("Age*")
        patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        patient_id = st.text_input("Patient ID*")
        
        # Medications
        medications = []
        for i in range(3):
            med_name = st.text_input(f"Medication {i+1} Name")
            med_dosage = st.text_input(f"Medication {i+1} Dosage")
            medications.append({
                'name': med_name, 
                'dosage': med_dosage,
                'frequency': "Once a day",
                'duration': "7 days",
                'special_instructions': "Take after meals"
            })
        
        # Submit button
        submitted = st.form_submit_button("Generate Prescription")
        
        if submitted:
            if patient_name and patient_age and patient_id:
                with st.spinner("Generating prescription..."):
                    patient_info = {
                        'name': patient_name,
                        'age': patient_age,
                        'gender': patient_gender,
                        'id': patient_id
                    }
                    
                    doctor_info = {
                        'name': "Dr. John Doe",
                        'specialty': "Orthopedics",
                        'license': "MED123456",
                        'contact': "(123) 456-7890"
                    }
                    
                    # Generate prescription PDF with fracture details
                    pdf_path = create_prescription(
                        patient_info=patient_info,
                        diagnosis="Fracture detected in the left arm.",
                        medications=medications,
                        instructions="Consult with a specialist.",
                        doctor_info=doctor_info,
                        fracture_image=st.session_state.fracture_image,
                        confidence_percentage=st.session_state.confidence_percentage,
                        model_name=st.session_state.selected_model_name
                    )
                    
                    st.success("Prescription generated successfully!")
                    st.markdown(create_download_link(pdf_path, "Medical_Prescription.pdf"), unsafe_allow_html=True)

if st.session_state.current_page == 'fracture_detection':
    show_fracture_detection()
else:
    show_prescription_generator()
