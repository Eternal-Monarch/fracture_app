import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
from fpdf import FPDF
import time
from streamlit.components.v1 import html

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

# Streamlit Page Configuration
st.set_page_config(
    page_title="BoneScan AI - Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CSS and THEMING
# =============================================
def inject_css():
    st.markdown("""
    <style>
    /* Base Styles */
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
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Layout */
    [data-testid="stAppViewContainer"] {
        background-color: var(--background);
        color: var(--text);
    }
    
    /* Cards */
    .card {
        background-color: var(--card-bg);
        color: var(--text);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    .stFileUploader>div {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        background-color: var(--card-bg);
    }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply initial theme
inject_css()

# =============================================
# SIDEBAR COMPONENTS
# =============================================
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=120)
    st.title("BoneScan AI")
    st.markdown("### Clinical Fracture Detection")
    
    # Model selection with enhanced UI
    selected_model_name = st.selectbox(
        "🧠 Select AI Model", 
        options=list(model_ids.keys()),
        help="Choose the deep learning model for analysis",
        index=0
    )

# =============================================
# PDF REPORT GENERATION
# =============================================
def generate_pdf_report(image, result, confidence_percent, selected_model_name, patient_name, doctor_name):
    pdf = FPDF()
    pdf.add_page()

    # Add logo (replace with your logo file path)
    pdf.image('path_to_logo.png', 10, 8, 33)
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="BoneScan AI - Fracture Detection Report", ln=True, align='C')

    # Patient details
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, txt=f"Doctor's Name: {doctor_name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {time.strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(10)

    # Result section
    pdf.cell(200, 10, txt=f"Model Used: {selected_model_name}", ln=True)
    pdf.cell(200, 10, txt=f"Fracture Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Level: {confidence_percent:.1f}%", ln=True)

    # Medical Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 10, txt="This tool is for research and preliminary assessment only. Please consult a medical professional for a definitive diagnosis.")
    
    # Advices based on result
    pdf.ln(10)
    if result == "Fracture Detected":
        pdf.cell(200, 10, txt="⚠️ Advices:", ln=True)
        pdf.multi_cell(0, 10, txt="""- Immediate orthopedic consultation recommended
- Immobilize the affected area
- Avoid putting weight on the injured limb
- Apply ice to reduce swelling if necessary""")
    else:
        pdf.cell(200, 10, txt="✅ No Fracture Detected", ln=True)
        pdf.multi_cell(0, 10, txt="""- If pain persists, consult a healthcare provider
- Consider follow-up imaging if symptoms worsen
- Practice proper bone health with calcium and vitamin D""")
    
    return pdf.output(dest='S').encode('latin1')

# =============================================
# MAIN CONTENT
# =============================================
# Header with gradient
st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>BoneScan AI</h1>
        <h3>Advanced Fracture Detection System</h3>
    </div>
""", unsafe_allow_html=True)

# Main content area
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Upload section with improved UX
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
        patient_name = st.text_input("Enter Patient Name:")
        doctor_name = st.text_input("Enter Doctor's Name:")
        
        try:
            # Display uploaded image
            image_file = Image.open(uploaded_file).convert("RGB")
            st.image(image_file, caption="Uploaded X-ray", use_column_width=True)
            
            # Load model and make predictions
            model = load_tensorflow_model(model_ids[selected_model_name], selected_model_name.replace(" ", "_"))
            processed_image = preprocess_image_tf(image_file, model)
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

            result = "Fracture Detected" if confidence > 0.5 else "Normal"
            confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
            confidence_percent = confidence_score * 100

            # Display results
            st.markdown(f"### 📝 Analysis Results")
            st.markdown(f"**Fracture Status**: {result}")
            st.markdown(f"**Confidence Level**: {confidence_percent:.1f}%")

            # Generate PDF report
            pdf = generate_pdf_report(image_file, result, confidence_percent, selected_model_name, patient_name, doctor_name)

            # Provide download button
            st.download_button(
                label="📄 Download Report",
                data=pdf,
                file_name="bonescan_analysis_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error analyzing the image: {str(e)}")

# Footer
st.markdown("""
    <div style="text-align: center; padding: 1rem 0; font-size: 0.9rem; color: #888;">
        <p>BoneScan AI v1.0 | Clinical Decision Support | © 2025 Medical AI Research Group</p>
    </div>
""", unsafe_allow_html=True)
