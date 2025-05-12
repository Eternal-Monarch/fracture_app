import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
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
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'fracture_detection'
if 'fracture_results' not in st.session_state:
    st.session_state.fracture_results = None

# Sidebar Navigation
with st.sidebar:
    st.title("BoneScan AI")
    st.markdown("---")
    
    # Navigation buttons
    if st.button("🦴 Fracture Detection"):
        st.session_state.current_page = 'fracture_detection'
    if st.button("💊 Generate Prescription"):
        st.session_state.current_page = 'prescription'
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("AI-powered fracture detection and prescription system for medical professionals.")
    
    if st.session_state.current_page == 'fracture_detection':
        selected_model_name = st.selectbox(
            "Select AI Model", 
            options=list(model_ids.keys())
    else:
        st.markdown("### Instructions")
        st.markdown("1. Fill patient details\n2. Add diagnosis\n3. Prescribe medications\n4. Generate PDF")

# Fracture Detection Page
def show_fracture_detection():
    st.title("🦴 Bone Fracture Detection")
    st.markdown("Upload an X-ray image to detect potential fractures.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
            
        with col2:
            if selected_model_name:
                with st.spinner(f"Loading {selected_model_name}..."):
                    file_id = model_ids[selected_model_name]
                    model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
                    
                with st.spinner("Analyzing image..."):
                    image = Image.open(uploaded_file).convert("RGB")
                    processed_image = preprocess_image_tf(image, model)
                    prediction = model.predict(processed_image)
                    confidence = prediction[0][0]
                    
                    result = "Fracture Detected" if confidence > 0.5 else "Normal"
                    confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
                    confidence_percent = confidence_score * 100
                    
                    # Store results in session state
                    st.session_state.fracture_results = {
                        'result': result,
                        'confidence': confidence_percent,
                        'model_name': selected_model_name,
                        'image': uploaded_file.getvalue()
                    }
                    
                    st.success("Analysis complete!")
                    st.markdown(f"**Result:** {result}")
                    st.markdown(f"**Confidence:** {confidence_percent:.1f}%")
                    st.markdown(f"**Model Used:** {selected_model_name}")
                    
                    # Visual indicator
                    st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <div style="display: flex; justify-content: space-between;">
                                <span>0%</span>
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                            <div style="height: 20px; background: linear-gradient(90deg, #ff6b6b, #51cf66); border-radius: 10px;">
                                <div style="height: 100%; width: {100 - confidence_percent}%; 
                                          background-color: #f8f9fa; border-radius: 10px; float: right;"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                        **Disclaimer:** This AI analysis is for preliminary assessment only. 
                        Final diagnosis must be made by a qualified medical professional.
                    """)

# Prescription Generator Page
def show_prescription_generator():
    st.title("💊 Medical Prescription Generator")
    
    # Show AI analysis results if available
    if st.session_state.fracture_results:
        with st.expander("AI Fracture Analysis Results", expanded=True):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(st.session_state.fracture_results['image'], 
                       caption="Analyzed X-ray", 
                       use_column_width=True)
                
            with col2:
                st.markdown(f"""
                    **AI Analysis Summary**  
                    - **Model Used:** {st.session_state.fracture_results['model_name']}  
                    - **Result:** {'<span style="color:red;font-weight:bold">Fracture Detected</span>' 
                                  if st.session_state.fracture_results['result'] == 'Fracture Detected' 
                                  else '<span style="color:green;font-weight:bold">No Fracture Detected</span>'}  
                    - **Confidence:** {st.session_state.fracture_results['confidence']:.1f}%  
                    
                    *Disclaimer: This AI analysis is for preliminary assessment only.  
                    Final diagnosis must be made by a qualified medical professional.*
                """, unsafe_allow_html=True)
    
    # Prescription form
    with st.form("prescription_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name*")
            patient_age = st.text_input("Age*")
            
        with col2:
            patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
            patient_id = st.text_input("Patient ID*")
        
        patient_allergies = st.text_area("Known Allergies", "None")
        
        # Pre-fill diagnosis if AI results available
        if st.session_state.fracture_results:
            default_diagnosis = f"AI-assisted analysis indicates: {st.session_state.fracture_results['result']} (Confidence: {st.session_state.fracture_results['confidence']:.1f}%)\n\n"
        else:
            default_diagnosis = ""
            
        diagnosis = st.text_area("Diagnosis*", value=default_diagnosis)
        
        st.subheader("Prescribed Medications")
        
        medications = []
        for i in range(2):  # Allow 2 medications
            with st.expander(f"Medication {i+1}", expanded=(i==0)):
                med_name = st.text_input(f"Name {i+1}", key=f"med_name_{i}")
                med_dosage = st.text_input(f"Dosage {i+1}", key=f"med_dosage_{i}")
                med_frequency = st.text_input(f"Frequency {i+1}", key=f"med_freq_{i}")
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
        
        instructions = st.text_area("Additional Instructions")
        
        st.subheader("Doctor Information")
        doctor_name = st.text_input("Doctor Name*")
        doctor_specialty = st.text_input("Specialty*")
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

# Main App Logic
if st.session_state.current_page == 'fracture_detection':
    show_fracture_detection()
else:
    show_prescription_generator()
