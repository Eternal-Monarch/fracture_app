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

# Model mappings
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "10B53bzc1pYrQnBfDqBWrDpNmzWoOl9ac",
    "MobileNet (Keras)": "1mlfoy6kKXUwIciZW3nftmiMHOTzpy6_s",
    "EfficientNetB3 (Keras)": "1cQA3_oH2XjDFK-ZE9D9YsP6Ya8fQiPOy"
}

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

def create_prescription(patient_info, diagnosis, medications, instructions, doctor_info):
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
    
    # Save PDF
    pdf_path = "medical_prescription.pdf"
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
    page_title="BoneScan AI - Medical Prescription",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
:root {
    --primary: #4E6EAF;
    --primary-dark: #3A5A8A;
    --secondary: #FF7E5D;
    --accent: #6C63FF;
    --background: #F8FAFC;
    --text: #2D3748;
    --text-light: #718096;
    --card-bg: #FFFFFF;
    --danger: #E53E3E;
    --success: #38A169;
    --warning: #DD6B20;
    --sidebar-bg: #FFFFFF;
    --border: #E2E8F0;
    --highlight: rgba(78, 110, 175, 0.1);
}

[data-theme="dark"] {
    --primary: #6C63FF;
    --primary-dark: #564EC2;
    --secondary: #FF7E5D;
    --accent: #4E6EAF;
    --background: #121212;
    --text: #E2E8F0;
    --text-light: #A0AEC0;
    --card-bg: #1E1E1E;
    --danger: #FC8181;
    --success: #68D391;
    --warning: #F6AD55;
    --sidebar-bg: #1A1A1A;
    --border: #2D3748;
    --highlight: rgba(108, 99, 255, 0.1);
}

/* Layout */
[data-testid="stAppViewContainer"] {
    background-color: var(--background);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border);
}

.main .block-container {
    padding: 2rem 2rem 1rem;
}

/* Cards */
.card {
    background-color: var(--card-bg);
    color: var(--text);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

.prescription-card {
    border-left: 4px solid var(--primary);
}

/* Form elements */
.stTextInput>div>div>input, 
.stTextArea>div>div>textarea,
.stSelectbox>div>div>select {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
}

/* Responsive */
@media (max-width: 768px) {
    .header h1 {
        font-size: 1.8rem;
    }
    
    .header h3 {
        font-size: 1rem;
    }
    
    .card {
        padding: 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)
    st.markdown("<h1 style='font-size: 1.8rem; margin-bottom: 0.5rem;'>BoneScan AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0;'>Medical Prescription System</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Fill patient information
    2. Enter diagnosis details
    3. Add prescribed medications
    4. Provide additional instructions
    5. Generate prescription
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown("""
    This tool assists in prescription creation only. 
    Final responsibility lies with the prescribing physician.
    """)

# Main Content
st.markdown("""
    <div style="background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                padding: 2rem;
                border-radius: 0 0 12px 12px;
                margin: -1rem -1rem 2rem -1rem;
                text-align: center;">
        <h1>Medical Prescription Generator</h1>
        <h3 style="font-weight: 400;">BoneScan AI Clinical System</h3>
    </div>
""", unsafe_allow_html=True)

# Main form
with st.form("prescription_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patient Information")
        patient_name = st.text_input("Full Name*")
        patient_age = st.text_input("Age*")
        patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
        patient_id = st.text_input("Patient ID*")
        
    with col2:
        st.markdown("### Medical Information")
        patient_allergies = st.text_area("Known Allergies", "None")
        diagnosis = st.text_area("Diagnosis*", placeholder="Primary diagnosis and relevant details")
        
    st.markdown("---")
    st.markdown("### Prescribed Medications")
    
    medications = []
    for i in range(3):  # Allow up to 3 medications
        with st.expander(f"Medication {i+1}", expanded=(i==0)):
            med_col1, med_col2, med_col3, med_col4 = st.columns(4)
            with med_col1:
                med_name = st.text_input(f"Name {i+1}", key=f"med_name_{i}")
            with med_col2:
                med_dosage = st.text_input(f"Dosage {i+1}", key=f"med_dosage_{i}")
            with med_col3:
                med_frequency = st.text_input(f"Frequency {i+1}", key=f"med_freq_{i}")
            with med_col4:
                med_duration = st.text_input(f"Duration {i+1}", key=f"med_dur_{i}")
            special_instructions = st.text_area(f"Special Instructions {i+1}", key=f"med_instr_{i}")
            
            if med_name and med_dosage:
                medications.append({
                    'name': med_name,
                    'dosage': med_dosage,
                    'frequency': med_frequency,
                    'duration': med_duration,
                    'special_instructions': special_instructions
                })
    
    st.markdown("---")
    st.markdown("### Additional Instructions")
    instructions = st.text_area("Patient instructions, follow-up details, etc.")
    
    st.markdown("---")
    st.markdown("### Physician Information")
    doc_col1, doc_col2 = st.columns(2)
    with doc_col1:
        doctor_name = st.text_input("Doctor Name*")
        doctor_specialty = st.text_input("Specialty*")
    with doc_col2:
        doctor_license = st.text_input("License Number*")
        doctor_contact = st.text_input("Contact Information*")
    
    submitted = st.form_submit_button("Generate Prescription")
    
    if submitted:
        if not all([patient_name, patient_age, patient_id, diagnosis, doctor_name, doctor_specialty, doctor_license]):
            st.error("Please fill all required fields (marked with *)")
        elif not medications:
            st.error("Please add at least one medication")
        else:
            with st.spinner("Generating prescription..."):
                patient_info = {
                    'name': patient_name,
                    'age': patient_age,
                    'gender': patient_gender,
                    'id': patient_id,
                    'allergies': patient_allergies
                }
                
                doctor_info = {
                    'name': doctor_name,
                    'specialty': doctor_specialty,
                    'license': doctor_license,
                    'contact': doctor_contact
                }
                
                pdf_path = create_prescription(
                    patient_info=patient_info,
                    diagnosis=diagnosis,
                    medications=medications,
                    instructions=instructions,
                    doctor_info=doctor_info
                )
                
                st.success("Prescription generated successfully!")
                st.markdown(create_download_link(pdf_path, "Medical_Prescription.pdf"), unsafe_allow_html=True)
                
                # Preview
                st.markdown("### Prescription Preview")
                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: var(--text-light); font-size: 0.9rem; padding: 1rem;">
        <p>BoneScan AI Medical Prescription System | Version 2.1</p>
        <p>© 2025 Radiology AI Research Group | NIT Meghalaya</p>
    </div>
""", unsafe_allow_html=True)
