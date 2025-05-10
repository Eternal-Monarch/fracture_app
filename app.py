


import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Model file paths
model_paths = {
    "DenseNet169 (Keras)": "models/normalbone_fracture_classifier_densenet169.keras",
    "InceptionV1 (Keras)": "models/normalbone_fracture_classifier_inceptionv3.keras",
    "MobileNetV3 with CNN (Keras)": "models/normalbone_fracture_classifier_mobilenet.keras",
    "EfficientNetB3 (Keras)": "models/efficientnet_b3_fracture_classifier.keras",
}


# Function to load models
@st.cache_resource  # Caches the model for performance
def load_tensorflow_model(path):
    try:
        return load_model(path)
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        st.stop()

# Preprocessing function (Automatically Resizes to Model Input Shape)
def preprocess_image_tf(uploaded_image, model):
    # Get the model's expected input shape (height, width)
    input_shape = model.input_shape[1:3]  # Extracts (height, width)

    img = uploaded_image.resize(input_shape).convert("L")  # Convert to grayscale
    
    # Convert grayscale to 3 channels (Replicating same values)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.stack([img_array] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

# Streamlit Page Configuration
st.set_page_config(page_title="Fracture Detection", page_icon="🦴", layout="wide")

# Sidebar - Theme & Model Selection
with st.sidebar:
    st.title("🔍 AI Fracture Classification")
    theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)

    selected_model_name = st.selectbox("Select a Model", options=list(model_paths.keys()))

    st.info("*Developer:* Subhasish Dutta")

# Dark Theme Styles
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: #ffffff; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .stFileUploader>div { background-color: #444444; border: 1px solid #555555; }
        </style>
    """, unsafe_allow_html=True)

# Light Theme Styles
else:
    st.markdown("""
        <style>
            body { background-color: #ffffff; color: #000000; }
            .stButton>button { background-color: #008CBA; color: white; }
            .stFileUploader>div { background-color: #f5f5f5; border: 1px solid #dddddd; }
        </style>
    """, unsafe_allow_html=True)

# Main Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Fracture Detection System</h1>", unsafe_allow_html=True)
st.write("📂 *Upload an X-ray image to detect fractures using AI.*")

# Load selected model
selected_model_path = model_paths[selected_model_name]
model = load_tensorflow_model(selected_model_path)

st.success(f"✅ {selected_model_name} loaded successfully!")

# File Upload
uploaded_file = st.file_uploader("📂 Upload an X-ray image...", type=["jpg", "jpeg", "png"])

# Process Image
if uploaded_file:
    try:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)  # Fixed deprecated parameter
        image_file = Image.open(uploaded_file).convert("RGB")
        processed_image = preprocess_image_tf(image_file, model)  # Fixed dynamic resizing issue

        # Prediction
        with st.spinner("🔍 Analyzing the image..."):
            prediction = model.predict(processed_image)
            confidence = prediction[0][0]

        result = "Fracture Detected" if confidence > 0.5 else "Normal"
        confidence_score = confidence if result == "Fracture Detected" else 1 - confidence

        st.subheader("🔎 Results")
        if result == "Fracture Detected":
            st.error(f"⚠ *{result}* with confidence: *{confidence_score:.2f}*")
            st.write("🔗 *Recommendation:* Consult a healthcare professional.")
        else:
            st.success(f"✅ *{result}* with confidence: *{confidence_score:.2f}*")
            st.write("🎉 *No fracture detected! Maintain healthy bones.*")
    
    except Exception as e:
        st.error(f"🚨 Error processing the image: {e}")

else:
    st.info("💡 Please upload an image to start the analysis.")
