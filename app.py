import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown
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

# Custom CSS (shared between themes)
st.markdown("""
    <style>
    /* Base styles */
    body {
        font-family: 'Inter', sans-serif;
    }
    
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 3rem 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none" opacity="0.1"><path d="M0,0 L100,0 L100,100 L0,100 Z" fill="white"/></svg>');
        background-size: 100px 100px;
        opacity: 0.1;
    }
    
    .header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        position: relative;
        font-weight: 700;
    }
    
    .header h3 {
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: 0;
        position: relative;
        opacity: 0.9;
    }
    
    .card {
        background-color: var(--card-bg);
        color: var(--text);
        border-radius: 12px;
        padding: 1.8rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.8rem;
        border: 1px solid var(--border);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    
    .model-card {
        border-left: 5px solid var(--primary);
    }
    
    .result-card {
        border-left: 5px solid var(--accent);
    }
    
    .upload-card {
        border-left: 5px solid var(--secondary);
    }
    
    .stProgress > div > div > div {
        background-color: var(--accent);
    }
    
    /* Button styles */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(108, 99, 255, 0.3);
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.4);
    }
    
    /* File uploader */
    .stFileUploader>div {
        border: 2px dashed var(--secondary);
        border-radius: 12px;
        padding: 3rem;
        background-color: var(--card-bg);
        transition: all 0.3s;
    }
    
    .stFileUploader>div:hover {
        border-color: var(--primary);
        background-color: var(--highlight);
    }
    
    /* Risk indicators */
    .risk-high {
        color: var(--danger);
        font-weight: bold;
        font-size: 1.3rem;
    }
    
    .risk-low {
        color: var(--success);
        font-weight: bold;
        font-size: 1.3rem;
    }
    
    .risk-medium {
        color: var(--warning);
        font-weight: bold;
        font-size: 1.3rem;
    }
    
    /* Confidence meter */
    .confidence-container {
        margin: 1.5rem 0;
    }
    
    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: var(--text);
        opacity: 0.8;
    }
    
    .confidence-meter {
        height: 12px;
        background: var(--border);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s ease-out;
        background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
    }
    
    .confidence-scale {
        display: flex;
        justify-content: space-between;
        color: var(--text);
        opacity: 0.6;
        font-size: 0.8rem;
    }
    
    /* Feature icons */
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1.2rem;
        color: var(--primary);
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Model performance indicators */
    .metric {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
    }
    
    .metric-bar {
        height: 8px;
        border-radius: 4px;
        margin-left: 1rem;
        flex-grow: 1;
        background-color: var(--border);
        position: relative;
    }
    
    .metric-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--card-bg);
        color: var(--text);
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        border: 1px solid var(--border);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-size: 0.9rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2rem;
        }
        
        .header h3 {
            font-size: 1rem;
        }
        
        .card {
            padding: 1.2rem;
        }
    }
    
    /* Fix for selectbox text color */
    .st-b7, .st-c0, .st-c1, .st-c2 {
        color: var(--text) !important;
    }
    
    /* Fix for radio button colors */
    .st-cf, .st-cg, .st-ch {
        background-color: var(--card-bg) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--border);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }
    </style>
""", unsafe_allow_html=True)
