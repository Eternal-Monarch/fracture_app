import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from fpdf import FPDF
from datetime import datetime
import base64

# Model mappings for fracture detection with local paths
# model_ids = {
#     "DenseNet169 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\normalbone_fracture_classifier_densenet169.keras",
#     "InceptionV3 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\inception model with the save modle option witht eh split dataset.ipynb", 
#     "MobileNet (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\final_model (3).keras",
#     "EfficientNetB3 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\efficientnet_b3_fracture_classifier.keras"
# }


# Model file paths
model_paths = {
    "DenseNet169 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\normalbone_fracture_classifier_densenet169.keras",
    "InceptionV3 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\inception model with the save modle option witht eh split dataset.ipynb", 
    "MobileNet (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\final_model (3).keras",
    "EfficientNetB3 (Keras)": r"C:\Users\Subhasish Dutta\Desktop\New folder\efficientnet_b3_fracture_classifier.keras"
}
# Function to load fracture detection model from local path
def load_tensorflow_model(file_path, model_name):
    if not os.path.exists(file_path):
        st.error(f"Model file {model_name} not found at {file_path}")
        return None
    return load_model(file_path)

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
    page_title="BoneScan AI - Fracture Detection & Prescription",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configuration in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fracture_detection'

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    set_theme()

# Apply theme based on session state
def set_theme():
    if st.session_state.theme == 'dark':
        dark_theme()
    else:
        light_theme()

def dark_theme():
    st.markdown("""<style>
        /* Add dark theme CSS styling here */
    </style>""", unsafe_allow_html=True)

def light_theme():
    st.markdown("""<style>
        /* Add light theme CSS styling here */
    </style>""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("BoneScan AI")
    
    # Theme toggle button
    if st.button(f"🌙 Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
        toggle_theme()
    
    st.markdown("---")
    
    # Navigation buttons
    st.markdown("### Navigation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🦴 Fracture Detection"):
            st.session_state.current_page = 'fracture_detection'
    with col2:
        if st.button("💊 Prescription"):
            st.session_state.current_page = 'prescription'
    
    st.markdown("---")
    
    if st.session_state.current_page == 'fracture_detection':
        selected_model_name = st.selectbox(
            "🧠 Select AI Model", 
            options=list(model_ids.keys()),
            help="Choose the deep learning model for analysis"
        )
        
        st.markdown("---")
        st.markdown("### 🔍 About")
        st.markdown("""
        BoneScan AI uses advanced deep learning to detect fractures in X-ray images. 
        This tool assists medical professionals in preliminary diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload a clear X-ray image
        2. Select analysis model
        3. View detailed results
        """)
    else:
        st.markdown("### 📝 Prescription Instructions")
        st.markdown("""
        1. Fill patient information
        2. Enter diagnosis details
        3. Add prescribed medications
        4. Provide additional instructions
        5. Generate prescription
        """)

# Main App Logic
if st.session_state.current_page == 'fracture_detection':
    show_fracture_detection()
else:
    show_prescription_generator()
