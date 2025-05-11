import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
from fpdf import FPDF
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

# Streamlit Page Configuration
st.set_page_config(
    page_title="BoneScan AI - Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configuration in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

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
                --primary: #6C63FF;
                --secondary: #4D44DB;
                --accent: #FF6584;
                --background: #121212;
                --text: #E0E0E0;
                --card-bg: #1E1E1E;
                --danger: #FF5252;
                --success: #4CAF50;
                --warning: #FFAB00;
                --sidebar-bg: #1A1A1A;
                --border: #333333;
                --highlight: rgba(108, 99, 255, 0.1);
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
                --primary: #6C63FF;
                --secondary: #4D44DB;
                --accent: #FF6584;
                --background: #F9F9F9;
                --text: #333333;
                --card-bg: #FFFFFF;
                --danger: #FF5252;
                --success: #4CAF50;
                --warning: #FFAB00;
                --sidebar-bg: #FFFFFF;
                --border: #E0E0E0;
                --highlight: rgba(108, 99, 255, 0.1);
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

# Apply initial theme
set_theme()

# Function to generate PDF report
def generate_pdf_report(image, result, confidence_percent, selected_model_name):
    pdf = FPDF()
    pdf.add_page()
    
    # Add logo (replace with your logo file path)
    pdf.image('path_to_logo.png', 10, 8, 33)
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="BoneScan AI - Fracture Detection Report", ln=True, align='C')

    # Result section
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"Model Used: {selected_model_name}", ln=True)
    pdf.cell(200, 10, txt=f"Fracture Status: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Level: {confidence_percent:.1f}%", ln=True)

    # Medical Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 10, txt="This tool is for research and preliminary assessment only. Please consult a medical professional for a definitive diagnosis.")

    return pdf.output(dest='S').encode('latin1')

# Sidebar and Main Content
st.markdown("""
    <div class="header">
        <h1 style="text-align: center; margin-bottom: 0.5rem;">🦴 BoneScan AI</h1>
        <h3 style="text-align: center; font-weight: 300; margin-top: 0;">
            Advanced Fracture Detection System
        </h3>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image_file = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

        # Load selected model
        with st.spinner(f"🔄 Loading {selected_model_name}..."):
            file_id = model_ids[selected_model_name]
            model = load_tensorflow_model(file_id, selected_model_name.replace(" ", "_"))
        
        # Preprocess and analyze the image
        processed_image = preprocess_image_tf(image_file, model)
        prediction = model.predict(processed_image)
        confidence = prediction[0][0]

        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_score = confidence if result == "Fracture Detected" else 1 - confidence
        confidence_percent = confidence_score * 100

        st.markdown(f"### 📝 Analysis Results")
        st.markdown(f"**Fracture Status**: {result}")
        st.markdown(f"**Confidence Level**: {confidence_percent:.1f}%")

        # Generate PDF report
        pdf = generate_pdf_report(image_file, result, confidence_percent, selected_model_name)

        # Provide download button
        st.download_button(
            label="📄 Download Report",
            data=pdf,
            file_name="bonescan_analysis_report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error analyzing the image: {str(e)}")
