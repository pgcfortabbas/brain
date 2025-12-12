import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
# NEW: Use huggingface_hub for simple access to files
from huggingface_hub import hf_hub_download

# --- Hugging Face Configuration ---
HF_REPO_ID = 'your-username/brain-tumor-model' # Replace with your actual HF repo ID
HF_FILENAME = 'brain_tumor_cnn_model.h5' 
MODEL_PATH = HF_FILENAME 

# --- Load the trained model ---
@st.cache_resource
def load_model():
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        st.warning(f"Model file not found. Attempting to download from Hugging Face Hub...")
        try:
            # Simple one-line download from HF
            hf_hub_download(
                repo_id=HF_REPO_ID, 
                filename=HF_FILENAME, 
                local_dir='.', # Download to the current directory
                local_dir_use_symlinks=False
            )
            st.success("Model downloaded successfully from Hugging Face!")
        except Exception as e:
            st.error(f"Error during model download from Hugging Face: {e}")
            return None

    # Load the model
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Troubleshooting: Model file is present but corrupted/invalid.")
        return None

# ... (The rest of your script: preprocess_image and Streamlit UI remains the same)
