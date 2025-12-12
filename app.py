import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
# NEW: Import 'requests' for handling HTTP requests (i.e., downloading the model file)
import requests

# --- Google Drive Configuration ---
# File ID extracted from the provided URL: 1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE
GDRIVE_FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
MODEL_PATH = 'brain_tumor_cnn_model.keras' # Local file path for the downloaded model

# --- Utility Function to Download from Google Drive ---
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from a public Google Drive link, handling the virus scan warning.
    """
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check for the confirmation token (for files over 100MB)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If a token is found, resend the request with the confirmation parameter
    if token:
        st.info("Bypassing Google Drive large file warning...")
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Raise an exception for bad status codes (e.g., 404, 403)
    response.raise_for_status()

    # Get total file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192 # 8 KiB chunks
    
    st.info(f"Downloading model (approx. {total_size / (1024*1024):.2f} MB)...")
    progress_bar = st.progress(0)

    # Download the file with streaming
    try:
        with open(destination, 'wb') as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Update progress bar
                    progress_bar.progress(min(100, int((downloaded_size / total_size) * 100)))
        
        progress_bar.progress(100) # Ensure it reaches 100%
        st.success("Model downloaded successfully!")

    except Exception as e:
        # Clean up the file if the download was interrupted
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
    
    # Check if the model is already downloaded, if not, download it
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Attempting to download from Google Drive...")
        try:
            download_file_from_google_drive(GDRIVE_FILE_ID, model_path)
        except requests.exceptions.HTTPError as e:
            st.error(f"Error downloading model: HTTP Error {e.response.status_code}. Please check the file ID and sharing permissions.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during download: {e}")
            return None

    # Load the model
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            # Ensure safe_mode=False might be needed for Keras H5 format or custom objects
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
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
    st.warning("Model could not be loaded. Please ensure you have the `requests` library installed (`pip install requests`) and the Google Drive link is public.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make prediction
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

# Ensure you have 'requests' installed: pip install requests
