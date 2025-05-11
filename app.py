import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image  # Only using Pillow (PIL)
import os
import gdown

# Mapping of model names to Google Drive file IDs
model_ids = {
    "DenseNet169 (Keras)": "1dIhc-0vd9sDoU5O6H0ZE6RYrP-CAyWks",
    "InceptionV3 (Keras)": "10B53bzc1pYrQnBfDqBWrDpNmzWoOl9ac",
    "MobileNet (Keras)": "14YuV3qZb_6FI7pXoiJx69HxiDD4uNc_Q",
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

# Preprocessing function using only Pillow (PIL)
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
        </style>
    """, unsafe_allow_html=True)

# Apply initial theme
set_theme()

# Custom CSS for overall styling
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
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nitm.ac.in/cygnus/nitmeghalaya/ckfinder/userfiles/images/NITM.gif", width=100)
    st.title("BoneScan AI")
    
    # Theme toggle button
    if st.button(f"🌙 Switch to {'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
        toggle_theme()
    
    st.markdown("---")
    
    selected_model_name = st.selectbox(
        "🧠 Select AI Model", 
        options=list(model_ids.keys()),
        help="Choose the deep learning model for analysis"
    )
    
    st.markdown("---")
    st.markdown("### 🔍 About")
    st.markdown("""BoneScan AI uses advanced deep learning to detect fractures in X-ray images. This tool assists medical professionals in preliminary diagnosis.""")

# Main Content
# Header Section
st.markdown("""
    <div class="header">
        <h1 style="text-align: center; margin-bottom: 0.5rem;">🦴 BoneScan AI</h1>
        <h3 style="text-align: center; font-weight: 300; margin-top: 0;">Advanced Fracture Detection System</h3>
    </div>
""", unsafe_allow_html=True)

# Main content columns
main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    st.markdown("""<div class="card upload-card"><h2>📤 Upload X-ray Image</h2><p>For best results, use clear, high-contrast images of the affected area.</p></div>""", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file:
        try:
            image_file = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True, output_format="PNG")
            
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
                    st.markdown("""<div class="card"><h3>⚠️ Medical Recommendation</h3><ul><li>Consult an orthopedic specialist immediately</li><li>Immobilize the affected area</li><li>Avoid putting weight on the injured limb</li><li>Apply ice to reduce swelling if appropriate</li></ul></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="card"><h3>✅ No Fracture Detected</h3><p>Our analysis found no evidence of fracture. Please monitor for any further symptoms.</p></div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error analyzing the image: {str(e)}")
