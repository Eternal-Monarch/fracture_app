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

@st.cache_resource
def load_tensorflow_model(file_id, model_name):
    model_path = f"models/{model_name}.keras"
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return load_model(model_path)

def preprocess_image_tf(uploaded_image, model):
    input_shape = model.input_shape[1:3]
    img = uploaded_image.resize(input_shape).convert("L")
    img_array = np.array(img) / 255.0
    img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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
    
    # Disclaimer
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report was generated by BoneScan AI and should be interpreted by a qualified medical professional. "
                         "The findings are based on available data and should be correlated with clinical examination.", 0, 'C')
    
    pdf_path = "medical_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def create_download_link(pdf_path, filename):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# Initialize session state
if 'prescription_data' not in st.session_state:
    st.session_state.prescription_data = {
        'patient_info': None,
        'diagnosis': '',
        'medications': [],
        'instructions': '',
        'doctor_info': None
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="BoneScan AI Medical System",
    page_icon="🏥",
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

.model-card {
    border-left: 4px solid var(--primary);
}

.result-card {
    border-left: 4px solid var(--secondary);
}

.upload-card {
    border-left: 4px solid var(--accent);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 20px;
    background-color: var(--card-bg);
    border-radius: 8px 8px 0 0;
    border: 1px solid var(--border);
}

.stTabs [aria-selected="true"] {
    background-color: var(--primary);
    color: white;
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
    st.markdown("<p style='font-size: 0.9rem; opacity: 0.8; margin-top: 0;'>Medical Imaging & Prescription System</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
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
        <h3 style="font-weight: 400;">X-ray Analysis & Prescription Tool</h3>
    </div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2 = st.tabs(["X-ray Analysis", "Prescription Generator"])

with tab1:
    st.markdown("""
        <div class="card upload-card">
            <h2>🦴 X-ray Image Analysis</h2>
            <p>Upload clear anterior-posterior or lateral view X-rays for fracture detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Select or drag X-ray image here (JPG/PNG)", 
            type=["jpg", "jpeg", "png"],
            key="xray_uploader"
        )
        
        if uploaded_file:
            try:
                # Save uploaded image
                image_path = "temp_xray.jpg"
                image_file = Image.open(uploaded_file).convert("RGB")
                image_file.save(image_path)
                
                st.image(image_file, caption="Uploaded X-ray", use_column_width=True)
                
                # Model selection
                selected_model_name = st.selectbox(
                    "🧠 Select AI Model", 
                    options=list(model_ids.keys()),
                    index=0,
                    key="model_select"
                )
                
                # Analyze button
                if st.button("Analyze X-ray", key="analyze_btn"):
                    with st.spinner(f"Loading {selected_model_name} model..."):
                        file_id = model_ids[selected_model_name]
                        model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
                        
                    with st.spinner("Analyzing X-ray..."):
                        processed_image = preprocess_image_tf(image_file, model)
                        prediction = model.predict(processed_image)
                        confidence = prediction[0][0]
                        result = "Fracture Detected" if confidence > 0.5 else "No Fracture Detected"
                        confidence_percent = confidence * 100 if result == "Fracture Detected" else (1 - confidence) * 100
                        
                        # Generate recommendations
                        if result == "Fracture Detected":
                            severity = "high" if confidence_percent > 75 else "medium" if confidence_percent > 50 else "low"
                            recommendations = [
                                f"{'Immediate' if severity == 'high' else 'Prompt'} orthopedic consultation",
                                "Immobilization of affected area",
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
                        st.markdown(f"""
                            <div class="card result-card">
                                <h2>Analysis Results</h2>
                                <div style="font-size: 1.2rem; margin: 1rem 0;">
                                    <strong>Diagnosis:</strong> <span class="{'risk-high' if result == 'Fracture Detected' else 'risk-low'}">
                                        {result}
                                    </span>
                                </div>
                                <div style="font-size: 1.2rem;">
                                    <strong>Confidence:</strong> {confidence_percent:.1f}%
                                </div>
                                <div style="font-size: 1.2rem; margin-top: 1rem;">
                                    <strong>Model:</strong> {selected_model_name}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Store data for prescription
                        st.session_state.prescription_data['diagnosis'] = result
                        
                        # Show button to generate prescription
                        if st.button("Generate Prescription Based on Results"):
                            st.session_state.active_tab = "Prescription Generator"
                            st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.markdown("""
                    <div class="card" style="border-left: 4px solid var(--danger);">
                        <h3>Processing Error</h3>
                        <p>Please try again with a different image or model.</p>
                        <p>Technical details: {}</p>
                    </div>
                """.format(str(e)), unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="card model-card">
                <h2>AI Model Information</h2>
                <p>Select a deep learning model for fracture detection:</p>
                
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
            </div>
            
            <div class="card">
                <h2>Imaging Guidelines</h2>
                <ul>
                    <li>Use proper anatomical positioning</li>
                    <li>Ensure adequate penetration</li>
                    <li>Include joint above/below injury</li>
                    <li>Minimize motion artifacts</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class="card upload-card">
            <h2>💊 Medical Prescription Generator</h2>
            <p>Create professional prescriptions for your patients</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("prescription_form"):
        st.markdown("### Patient Information")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            patient_name = st.text_input("Full Name*", key="p_name_rx")
            patient_age = st.text_input("Age*", key="p_age_rx")
        with p_col2:
            patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"], key="p_gender_rx")
            patient_id = st.text_input("Patient ID*", key="p_id_rx")
        patient_allergies = st.text_area("Known Allergies", "None", key="p_allergies_rx")
        
        st.markdown("### Clinical Information")
        diagnosis = st.text_area("Diagnosis*", 
                               value=st.session_state.prescription_data.get('diagnosis', ''),
                               placeholder="Primary diagnosis and relevant details", 
                               key="diagnosis_rx")
        
        st.markdown("### Prescribed Medications")
        medications = []
        for i in range(3):  # Allow up to 3 medications
            with st.expander(f"Medication {i+1}", expanded=(i==0)):
                med_col1, med_col2, med_col3, med_col4 = st.columns(4)
                with med_col1:
                    med_name = st.text_input(f"Name {i+1}", key=f"med_name_{i}_rx")
                with med_col2:
                    med_dosage = st.text_input(f"Dosage {i+1}", key=f"med_dosage_{i}_rx")
                with med_col3:
                    med_frequency = st.text_input(f"Frequency {i+1}", key=f"med_freq_{i}_rx")
                with med_col4:
                    med_duration = st.text_input(f"Duration {i+1}", key=f"med_dur_{i}_rx")
                special_instructions = st.text_area(f"Special Instructions {i+1}", key=f"med_instr_{i}_rx")
                
                if med_name and med_dosage:
                    medications.append({
                        'name': med_name,
                        'dosage': med_dosage,
                        'frequency': med_frequency,
                        'duration': med_duration,
                        'special_instructions': special_instructions
                    })
        
        st.markdown("### Additional Instructions")
        instructions = st.text_area("Patient instructions, follow-up details, etc.", key="instructions_rx")
        
        st.markdown("### Physician Information")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            doctor_name = st.text_input("Doctor Name*", key="d_name_rx")
            doctor_specialty = st.text_input("Specialty*", key="d_specialty_rx")
        with d_col2:
            doctor_license = st.text_input("License Number*", key="d_license_rx")
            doctor_contact = st.text_input("Contact Information*", key="d_contact_rx")
        
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
                    
                    # Store data in session state
                    st.session_state.prescription_data = {
                        'patient_info': patient_info,
                        'diagnosis': diagnosis,
                        'medications': medications,
                        'instructions': instructions,
                        'doctor_info': doctor_info
                    }
                    
                    pdf_path = create_medical_report(
                        report_type="Prescription",
                        image_path=None,
                        result=None,
                        confidence=None,
                        model_name=None,
                        recommendations=None,
                        patient_info=patient_info,
                        medications=medications,
                        instructions=instructions,
                        doctor_info=doctor_info
                    )
                    
                    st.success("Prescription generated successfully!")
                    st.markdown(create_download_link(pdf_path, "Medical_Prescription.pdf"), unsafe_allow_html=True)
                    
                    # Preview
                    with open(pdf_path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: var(--text-light); font-size: 0.9rem; padding: 1rem;">
        <p>BoneScan AI Medical System | Version 2.1</p>
        <p>© 2025 Radiology AI Research Group | NIT Meghalaya</p>
    </div>
""", unsafe_allow_html=True)
