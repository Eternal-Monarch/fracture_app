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
    st.markdown(f"""
        <style>
            :root {{
                --primary: #4a8fe7;
                --secondary: #2d3748;
                --accent: #44e5e7;
                --background: #1a202c;
                --text: #e2e8f0;
                --card-bg: #2d3748;
                --danger: #fc8181;
                --success: #68d391;
                --sidebar-bg: #1a202c;
                --border: #4a5568;
            }}
            
            [data-testid="stAppViewContainer"] {{
                background-color: var(--background);
                color: var(--text);
            }}
            
            [data-testid="stSidebar"] {{
                background-color: var(--sidebar-bg) !important;
                border-right: 1px solid var(--border);
            }}
            
            .st-b7 {{
                color: var(--text) !important;
            }}
            
            .stFileUploader>div {{
                background-color: var(--card-bg) !important;
                border-color: var(--border) !important;
            }}
            
            .css-1aumxhk {{
                color: var(--text);
            }}
        </style>
    """, unsafe_allow_html=True)

def light_theme():
    st.markdown(f"""
        <style>
            :root {{
                --primary: #4a8fe7;
                --secondary: #c1d3fe;
                --accent: #44e5e7;
                --background: #f8f9fa;
                --text: #333333;
                --card-bg: #ffffff;
                --danger: #ff6b6b;
                --success: #51cf66;
                --sidebar-bg: #f8f9fa;
                --border: #e2e8f0;
            }}
        </style>
    """, unsafe_allow_html=True)

# Apply initial theme
set_theme()

# Custom CSS (shared between themes)
st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white;
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .card {
        background-color: var(--card-bg);
        color: var(--text);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }
    
    .model-card {
        border-left: 4px solid var(--primary);
    }
    
    .result-card {
        border-left: 4px solid var(--accent);
    }
    
    .upload-card {
        border-left: 4px solid var(--secondary);
    }
    
    .stProgress > div > div > div {
        background-color: var(--accent);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--accent);
        transform: translateY(-2px);
    }
    
    .stFileUploader>div {
        border: 2px dashed var(--secondary);
        border-radius: 10px;
        padding: 2rem;
        background-color: var(--card-bg);
    }
    
    .risk-high {
        color: var(--danger);
        font-weight: bold;
    }
    
    .risk-low {
        color: var(--success);
        font-weight: bold;
    }
    
    .confidence-meter {
        height: 20px;
        background: linear-gradient(90deg, var(--danger), var(--success));
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background-color: var(--card-bg);
        border-radius: 10px;
        transition: width 0.5s;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    /* Navigation buttons */
    .nav-button {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        border-radius: 8px;
        background-color: var(--primary);
        color: white;
        text-decoration: none;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background-color: var(--accent);
        transform: translateY(-2px);
    }
    
    .nav-button.active {
        background-color: var(--accent);
        font-weight: bold;
    }
    
    /* Fix for selectbox text color */
    .st-b7, .st-c0, .st-c1, .st-c2 {
        color: var(--text) !important;
    }
    
    /* Fix for radio button colors */
    .st-cf, .st-cg, .st-ch {
        background-color: var(--card-bg) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=100)
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
    
    st.markdown("---")
    st.markdown("👨‍⚕ *Medical Disclaimer*")
    st.markdown("""
    This tool is for research purposes only. Always consult a qualified healthcare professional for medical diagnosis.
    """)

# Fracture Detection Page
def show_fracture_detection():
    # Header Section
    st.markdown("""
        <div class="header">
            <h1 style="text-align: center; margin-bottom: 0.5rem;">🦴 BoneScan AI</h1>
            <h3 style="text-align: center; font-weight: 300; margin-top: 0;">
                Advanced Fracture Detection System
            </h3>
        </div>
    """, unsafe_allow_html=True)

    # Three column layout for features
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="card" style="text-align: center;">
                <div class="feature-icon">⚡</div>
                <h3>Rapid Analysis</h3>
                <p>Get results in seconds with our optimized AI models</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="card" style="text-align: center;">
                <div class="feature-icon">🔍</div>
                <h3>Multi-Model</h3>
                <p>Choose from several state-of-the-art deep learning architectures</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="card" style="text-align: center;">
                <div class="feature-icon">📊</div>
                <h3>Detailed Reports</h3>
                <p>Comprehensive analysis with confidence metrics</p>
            </div>
        """, unsafe_allow_html=True)

    # Main content columns
    main_col1, main_col2 = st.columns([2, 1])

    with main_col1:
        st.markdown("""
            <div class="card upload-card">
                <h2>📤 Upload X-ray Image</h2>
                <p>For best results, use clear, high-contrast images of the affected area.</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                image_file = Image.open(uploaded_file).convert("RGB")
                st.image(
                    uploaded_file, 
                    caption="Uploaded X-ray", 
                    use_column_width=True,
                    output_format="PNG"
                )
                
                # Load selected model
                with st.spinner(f"🔄 Loading {selected_model_name}..."):
                    file_id = model_ids[selected_model_name]
                    model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
                    
                with st.spinner("🔍 Analyzing image..."):
                    processed_image = preprocess_image_tf(image_file, model)
                    prediction = model.predict(processed_image)
                    confidence = prediction[0][0]
                    
                    result = "Fracture Detected" if confidence > 0.5 else "Normal"
                    confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
                    confidence_percent = confidence_score * 100
                    
                    # Visualization
                    st.markdown(f"""
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: {100 - confidence_percent}%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; color: var(--text);">
                            <span>0%</span>
                            <span>50%</span>
                            <span>100%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Results card
                    st.markdown(f"""
                        <div class="card result-card">
                            <h2>📝 Analysis Results</h2>
                            <div style="font-size: 1.2rem; margin: 1rem 0;">
                                Status: <span class="{'risk-high' if result == 'Fracture Detected' else 'risk-low'}">
                                    {result}
                                </span>
                            </div>
                            <div style="font-size: 1.2rem;">
                                Confidence: <strong>{confidence_percent:.1f}%</strong>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Recommendations
                    if result == "Fracture Detected":
                        st.markdown("""
                            <div class="card" style="border-left: 4px solid var(--danger);">
                                <h3>⚠ Medical Recommendation</h3>
                                <p>Our analysis indicates a potential fracture. Please:</p>
                                <ul>
                                    <li>Consult an orthopedic specialist immediately</li>
                                    <li>Immobilize the affected area</li>
                                    <li>Avoid putting weight on the injured limb</li>
                                    <li>Apply ice to reduce swelling if appropriate</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="card" style="border-left: 4px solid var(--success);">
                                <h3>✅ No Fracture Detected</h3>
                                <p>Our analysis found no evidence of fracture. However:</p>
                                <ul>
                                    <li>If pain persists, consult a healthcare provider</li>
                                    <li>Consider follow-up imaging if symptoms worsen</li>
                                    <li>Practice proper bone health with calcium and vitamin D</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                        
            except Exception as e:
                st.error(f"Error analyzing the image: {str(e)}")

    with main_col2:
        st.markdown("""
            <div class="card model-card">
                <h2>🧠 Selected Model</h2>
                <p><strong>{}</strong></p>
                <p>This model analyzes bone structures to detect potential fractures with advanced computer vision techniques.</p>
            </div>
        """.format(selected_model_name), unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h2>ℹ How It Works</h2>
                <ol>
                    <li>Upload X-ray image</li>
                    <li>AI processes image features</li>
                    <li>Deep learning analysis</li>
                    <li>Confidence score generated</li>
                    <li>Results displayed</li>
                </ol>
                <p><small>Note: Analysis takes 10-30 seconds depending on model complexity.</small></p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="card">
                <h2>📊 Model Performance</h2>
                <p>Average metrics across validation set:</p>
                <ul>
                    <li>Accuracy: 92-96%</li>
                    <li>Sensitivity: 89-94%</li>
                    <li>Specificity: 93-97%</li>
                </ul>
                <p><small>Performance varies by model and image quality.</small></p>
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: var(--text); opacity: 0.7; font-size: 0.9rem; padding: 1rem;">
            <p>BoneScan AI v1.0 | For research purposes only | Not for clinical use</p>
            <p>© 2025 Medical AI Research Group | All rights reserved</p>
        </div>
    """, unsafe_allow_html=True)

# Prescription Generator Page
def show_prescription_generator():
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

# Main App Logic
if st.session_state.current_page == 'fracture_detection':
    show_fracture_detection()
else:
    show_prescription_generator()
