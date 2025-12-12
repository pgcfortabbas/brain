import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests 

# --- Google Drive Configuration ---
# *** UPDATED FILE ID ***
GDRIVE_FILE_ID = '12Zm-LJUT3_iQW-4Hu1SXnybgHyKXlo04'
# Using .h5 extension for maximum Keras compatibility
MODEL_PATH = 'model.keros' 

# --- Utility Function to Download from Google Drive (Robust version) ---
def download_file_from_google_drive(file_id, destination):
    """Downloads a file from a public Google Drive link, bypassing the large file warning."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    # Initial request to check for the warning token
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    params = {'id': file_id}
    
    # If a token is found, add the confirmation parameter
    if token:
        st.info("Bypassing Google Drive large file warning...")
        params['confirm'] = token
        
    # Final request for the actual file content
    response = session.get(URL, params=params, stream=True)
    
    # Check for HTML content (indicates failure)
    if 'text/html' in response.headers.get('content-type', ''):
        st.error("Download Failed: The response was HTML, not the model file.")
        st.error("Please check that the Google Drive file is set to 'Anyone with the link'.")
        raise Exception("Google Drive link is inaccessible or non-public.")

    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    st.info(f"Downloading model (approx. {total_size / (1024*1024):.2f} MB)...")
    progress_bar = st.progress(0)

    try:
        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Update progress bar
                    progress_bar.progress(min(100, int((downloaded_size / total_size) * 100)))
        
        # Check for zero-byte file
        if downloaded_size == 0:
            raise Exception("Zero-byte file downloaded. The link is likely invalid or inaccessible.")

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
    
    # Logic to check and download the file if it doesn't exist locally
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
            # Using safe_mode=False for compatibility with H5 format
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Troubleshooting: This error occurs when the downloaded file is corrupted. Please ensure the Google Drive file is **publicly accessible** (set to 'Anyone with the link').")
        return None

model = load_model()

# --- Define image preprocessing function ---
def preprocess_image(img_data):
    img = Image.open(img_data)
    img = img.resize((150, 150)) # Ensure this matches img_width, img_height from training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch dimension
    img_array = img_array / 255.0 # Rescale pixels to [0, 1] as done during training
    return img_array

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a brain MRI image to get a tumor classification prediction.")

if model is None:
    st.warning("Model could not be loaded. Please ensure the Google Drive file (`18LzThwAFDKJQH99vyYcJAtdUPyaHvCZ5`) is **publicly accessible**.")
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
        
        # Assuming class labels are in the same order as train_generator.class_indices
        class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.success(f"Prediction: **{predicted_class_label.replace('_', ' ').title()}**")
        st.info(f"Confidence: {confidence:.2f}%")

        st.markdown("---")
        st.markdown("### About the Model")
        st.markdown(
            "This model is a Convolutional Neural Network trained to classify brain MRI images into one of four categories: "
            "Glioma Tumor, Meningioma Tumor, No Tumor, or Pituitary Tumor."
        )
        st.markdown(
            "**Disclaimer**: This is a demo for educational purposes and should not be used for medical diagnosis."
        )
