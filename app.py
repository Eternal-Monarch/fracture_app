import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
from fpdf import FPDF
from datetime import datetime
import base64

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
    img = uploaded_image.resize(input_shape).convert("L")
    img_array = np.array(img) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# PDF Generation Class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'BoneScan AI Medical Report', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Function to create a medical report (either X-ray or Prescription)
def create_medical_report(report_type, image_path=None, result=None, confidence=None, 
                         model_name=None, recommendations=None, patient_info=None, 
                         medications=None, instructions=None, doctor_info=None):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Header with clinic info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 5, "BoneScan AI Medical Center", 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, "123 Medical Drive, Healthcare City", 0, 1, 'C')
    pdf.cell(0, 5, "Phone: (123) 456-7890 | License: MED123456", 0, 1, 'C')
    pdf.ln(5)
    
    # Report metadata
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"Report Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 0, 1, 'R')
    pdf.cell(0, 5, f"Report Type: {report_type}", 0, 1, 'R')
    pdf.ln(5)
    
    if report_type == "X-ray Analysis":
        # Image Section
        if image_path and os.path.exists(image_path):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "X-RAY IMAGE", 0, 1, 'C')
            try:
                pdf.image(image_path, x=50, w=110)
            except:
                pdf.cell(0, 10, "[Image could not be loaded]", 0, 1, 'C')
            pdf.ln(10)

        # Results Section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "ANALYSIS RESULTS", 0, 1, 'C')
        pdf.ln(5)

        pdf.set_fill_color(230, 230, 230)
        pdf.rect(20, pdf.get_y(), 170, 30, 'F')

        pdf.set_font('Arial', 'B', 12)
        pdf.set_xy(25, pdf.get_y() + 5)
        pdf.cell(40, 5, "Diagnosis:", 0, 0)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(220, 50, 50) if "Fracture" in result else pdf.set_text_color(50, 150, 50)
        pdf.cell(40, 5, result, 0, 1)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 11)
        pdf.set_xy(25, pdf.get_y() + 5)
        pdf.cell(40, 5, "Confidence Level:", 0, 0)
        pdf.cell(40, 5, f"{confidence:.1f}%", 0, 1)

        pdf.set_xy(25, pdf.get_y() + 5)
        pdf.cell(40, 5, "AI Model Used:", 0, 0)
        pdf.cell(40, 5, model_name, 0, 1)
        pdf.ln(15)
        
        # Recommendations
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "MEDICAL RECOMMENDATIONS", 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        for rec in recommendations:
            pdf.cell(10, 7)
            pdf.cell(0, 7, f"• {rec}", 0, 1)
    
    elif report_type == "Prescription":
        # Medications
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "PRESCRIBED MEDICATIONS", 0, 1, 'C')
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

    # Additional Instructions
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "ADDITIONAL INSTRUCTIONS", 0, 1, 'C')
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, instructions)
    pdf.ln(15)

    # Doctor information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "ATTENDING PHYSICIAN", 0, 1, 'C')
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

    # Save PDF
    pdf_path = "medical_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def create_download_link(pdf_path, filename):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Streamlit App Configuration
st.set_page_config(
    page_title="BoneScan AI - Fracture Detection & Medical Prescription",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)
    st.markdown("<h1 style='font-size: 1.8rem; margin-bottom: 0.5rem;'>BoneScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0;'>Medical Imaging & Prescription System</p>", unsafe_allow_html=True)
    
    selected_model_name = st.selectbox(
        "🧠 Select AI Model", 
        options=list(model_ids.keys())
    )
    
    st.markdown("---")
    st.markdown("### ⚠ Medical Disclaimer")
    st.markdown("""
    This tool assists medical professionals only. 
    Final diagnosis and treatment decisions must be made by qualified physicians.
    """)

# Main Content
st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                padding: 2rem;
                border-radius: 0 0 12px 12px;
                margin: -1rem -1rem 2rem -1rem;
                text-align: center;">
        <h1>BoneScan AI Medical System</h1>
        <h3 style="font-weight: 400;">Fracture Detection & Prescription Generation</h3>
    </div>
""", unsafe_allow_html=True)

# Tabs for X-ray Analysis and Prescription Generator
tab1, tab2 = st.tabs(["X-ray Analysis", "Prescription Generator"])

with tab1:
    st.markdown("""<div class="card upload-card"><h2>🦴 X-ray Image Analysis</h2></div>""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select or drag X-ray image here (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image_file = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
        
        # Load selected model
        file_id = model_ids[selected_model_name]
        model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
        
        # Analyze button
        if st.button("Analyze X-ray"):
            processed_image = preprocess_image_tf(image_file, model)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]
            result = "Fracture Detected" if confidence > 0.5 else "Normal"
            confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
            confidence_percent = confidence_score * 100
            st.markdown(f"### Analysis Result: {result} with {confidence_percent:.1f}% confidence")
            
            if result == "Fracture Detected":
                recommendations = [
                    "Consult an orthopedic specialist immediately",
                    "Immobilize the affected area",
                    "Pain management as needed",
                    "Follow-up imaging recommended"
                ]
            else:
                recommendations = [
                    "Clinical correlation if symptomatic",
                    "Consider follow-up if pain persists",
                    "RICE protocol as needed"
                ]
            
            # Display results
            st.markdown(f"### Recommendations: {', '.join(recommendations)}")

with tab2:
    st.markdown("""<div class="card upload-card"><h2>💊 Medical Prescription Generator</h2></div>""", unsafe_allow_html=True)
    
    with st.form("prescription_form"):
        patient_name = st.text_input("Full Name*")
        patient_age = st.text_input("Age*")
        patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        patient_id = st.text_input("Patient ID*")
        diagnosis = st.text_area("Diagnosis*")
        
        medications = []
        for i in range(3):
            with st.expander(f"Medication {i+1}", expanded=(i == 0)):
                med_name = st.text_input(f"Name {i+1}")
                med_dosage = st.text_input(f"Dosage {i+1}")
                med_frequency = st.text_input(f"Frequency {i+1}")
                med_duration = st.text_input(f"Duration {i+1}")
                med_special_instructions = st.text_area(f"Special Instructions {i+1}")
                
                if med_name and med_dosage:
                    medications.append({
                        "name": med_name,
                        "dosage": med_dosage,
                        "frequency": med_frequency,
                        "duration": med_duration,
                        "special_instructions": med_special_instructions
                    })
        
        if st.form_submit_button("Generate Prescription"):
            patient_info = {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "id": patient_id
            }
            doctor_info = {
                "name": "Dr. Smith",
                "specialty": "Orthopedic",
                "license": "12345",
                "contact": "123-456-7890"
            }
            
            pdf_path = create_medical_report(
                report_type="Prescription",
                result=None,
                confidence=None,
                model_name=None,
                recommendations=None,
                patient_info=patient_info,
                medications=medications,
                instructions="Follow the prescribed medication regimen carefully.",
                doctor_info=doctor_info
            )
            
            st.success("Prescription generated successfully!")
            st.markdown(create_download_link(pdf_path, "Medical_Prescription.pdf"), unsafe_allow_html=True)
            
            # Preview the PDF
            st.markdown(f'<iframe src="data:application/pdf;base64,{base64.b64encode(open(pdf_path, "rb").read()).decode("utf-8")}" width="100%" height="600"></iframe>', unsafe_allow_html=True)
