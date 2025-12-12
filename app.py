import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests 
from huggingface_hub import hf_hub_download # New dependency for reliable download

# --- Hugging Face Configuration ---
# IMPORTANT: Replace 'your-username/brain-tumor-model' with your actual Hugging Face repository ID
HF_REPO_ID = 'your-username/brain-tumor-model' 
HF_FILENAME = 'brain_tumor_cnn_model.h5' 
MODEL_PATH = HF_FILENAME 

# Set Streamlit page config
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Load the trained model ---
@st.cache_resource
def load_model():
    model_path = MODEL_PATH
    
    # Check if the model is downloaded, if not, download it from Hugging Face
    if not os.path.exists(model_path):
        st.warning(f"Model file not found. Attempting to download from Hugging Face Hub: {HF_REPO_ID}...")
        try:
            # Download the file using hf_hub_download
            hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=HF_FILENAME, 
                local_dir='.', # Download to the current directory
                local_dir_use_symlinks=False
            )
            st.success("Model downloaded successfully from Hugging Face!")
        except Exception as e:
            st.error(f"Error during model download from Hugging Face. Please check the repo ID and filename: {e}")
            # Fallback for visibility: Try to clean up a potentially partial file
            if os.path.exists(model_path):
                os.remove(model_path)
            return None

    # Load the model
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            # Using safe_mode=False for compatibility with H5 format
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Troubleshooting: Model file is present but corrupted or invalid. Ensure the file on Hugging Face is correct.")
        return None

model = load_model()

# --- Define image preprocessing function ---
def preprocess_image(img_data):
    img = Image.open(img_data)
    img = img.resize((150, 150)) # Input size from training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale
    return img_array

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a brain MRI image to get a tumor classification prediction.")

if model is None:
    st.warning("Model could not be loaded. Please ensure the model is uploaded correctly to the Hugging Face Hub, and the `requirements.txt` file is properly configured.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make prediction
        with st.spinner("Analyzing image..."):
            predictions = model.predict(processed_image)
        
        # Class labels
        class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.success(f"Prediction: **{predicted_class_label.replace('_', ' ').title()}**")
        st.info(f"Confidence: {confidence:.2f}%")

        st.markdown("---")
        st.markdown("### About the Model")
        
        st.markdown(
            """
            This model is a **Convolutional Neural Network (CNN)** trained to classify brain MRI images into one of four categories: 
            **Glioma Tumor**, **Meningioma Tumor**, **No Tumor**, or **Pituitary Tumor**.
            """
        )
        st.markdown(
            "**Disclaimer**: This is a demo for educational purposes and should not be used for medical diagnosis."
        )
