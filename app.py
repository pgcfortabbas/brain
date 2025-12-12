import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests # Ensure you have this installed: pip install requests

# --- Google Drive Configuration ---
GDRIVE_FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
# *** FIXED: Changed local file path to .h5 for compatibility ***
MODEL_PATH = 'brain_tumor_cnn_model.h5' # Local file path for the downloaded model

# --- Utility Function to Download from Google Drive ---
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from a public Google Drive link, handling the virus scan warning.
    (Function body remains the same as provided in the previous response)
    """
    # ... (Keep the rest of your download_file_from_google_drive function as is)
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        st.info("Bypassing Google Drive large file warning...")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    # *** Added check for 0 total size to catch failed downloads early ***
    if total_size == 0:
        st.error("Download failed: Content size is 0 bytes. Check if the link is public and valid.")
        raise Exception("Zero-byte file downloaded.")
        
    block_size = 8192
    
    st.info(f"Downloading model (approx. {total_size / (1024*1024):.2f} MB)...")
    progress_bar = st.progress(0)

    try:
        with open(destination, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress_bar.progress(min(100, int((downloaded_size / total_size) * 100)))
        
        progress_bar.progress(100)
        st.success("Model downloaded successfully!")

    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise e
    finally:
        progress_bar.empty()


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
    
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Attempting to download from Google Drive...")
        try:
            download_file_from_google_drive(GDRIVE_FILE_ID, model_path)
        except Exception as e:
            st.error(f"Error during model download: {e}")
            return None

    # Load the model
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            # *** FIXED: Using safe_mode=False for compatibility ***
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Troubleshooting: If the model is an H5 file, ensure the local path ends with '.h5'. If it is the Keras native format, ensure the file is an accessible zip archive.")
        return None

model = load_model()

# ... (The rest of your script for preprocess_image and Streamlit UI remains the same)
