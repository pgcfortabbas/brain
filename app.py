import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests 

# --- Google Drive Configuration ---
GDRIVE_FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
# Keep the .h5 extension as planned for compatibility
MODEL_PATH = 'brain_tumor_cnn_model.h5' 

# --- Utility Function to Download from Google Drive (MORE ROBUST) ---
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from a public Google Drive link, ensuring all parameters 
    are passed correctly to bypass the large file warning.
    """
    # The URL pattern for downloading Google Drive files
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    # 1. Initial request to check for the warning token
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    # Check for the confirmation token (for files over 100MB)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    params = {'id': file_id}
    
    # 2. Add the token to the parameters if found
    if token:
        st.info("Bypassing Google Drive large file warning...")
        params['confirm'] = token
        
    # 3. Final request for the actual file content
    response = session.get(URL, params=params, stream=True)
    
    # Check for HTML content, which indicates a failure
    if 'text/html' in response.headers.get('content-type', ''):
        st.error("Download Failed: The response was HTML, not the model file.")
        st.error("This usually means the Google Drive link is not public or requires login.")
        raise Exception("Google Drive link is inaccessible or non-public.")

    # Check for bad HTTP status codes (e.g., 404, 403)
    response.raise_for_status()

    # Get total file size for progress bar and sanity check
    total_size = int(response.headers.get('content-length', 0))
    
    # --- CRITICAL SANITY CHECK ---
    if total_size < 1024 * 1024 * 10: # Check if size is less than 10MB (A guess, but 0.00 MB is definitely wrong)
        # Note: We're not raising an error here, but we check for 0 bytes later
        pass

    block_size = 8192 # 8 KiB chunks
    
    st.info(f"Downloading model (approx. {total_size / (1024*1024):.2f} MB)...")
    progress_bar = st.progress(0)

    # Download the file with streaming
    try:
        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Update progress bar
                    progress_bar.progress(min(100, int((downloaded_size / total_size) * 100)))
        
        # --- CRITICAL ZERO-BYTE CHECK ---
        if downloaded_size == 0:
            raise Exception("Zero-byte file downloaded. The link is likely invalid or inaccessible.")

        progress_bar.progress(100) # Ensure it reaches 100%
        st.success("Model downloaded successfully!")

    except Exception as e:
        # Clean up the file if the download was interrupted or failed
        if os.path.exists(destination):
            os.remove(destination)
        raise e
    finally:
        progress_bar.empty()


# Set Streamlit page config (remaining part of the script)
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
        except Exception as e:
            st.error(f"Error during model download: {e}")
            return None

    # Load the model
    try:
        with st.spinner(f"Loading model from {model_path}..."):
            # Ensure safe_mode=False is used for H5 files
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Troubleshooting: Please ensure the Google Drive file is **publicly accessible** (set to 'Anyone with the link can view').")
        return None

model = load_model()

# --- Define image preprocessing function ---
def preprocess_image(img_data):
    # ... (Keep the preprocess_image function as is)
    img = Image.open(img_data)
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a brain MRI image to get a tumor classification prediction.")

if model is None:
    st.warning("Model could not be loaded. **Please check the Google Drive sharing settings for the file ID: 1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE**.")
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
