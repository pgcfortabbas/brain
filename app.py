import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests 

# --- Configuration ---
# File ID from the Google Drive link: 1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE
GDRIVE_FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
# We use .h5 for the local file path
MODEL_PATH = 'brain_tumor_cnn_model.h5' 

# --- Utility Function: Simplified Raw Download from Google Drive ---
def download_file_from_google_drive_raw(file_id, destination):
    """Downloads the file using the direct export URL, bypassing token checks."""
    
    # Use the raw export URL
    URL = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    st.info("Attempting simplified raw download...")
    
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Basic check for HTML/error pages
        if 'text/html' in response.headers.get('content-type', '').lower():
            st.error("Download Failed: Received HTML response. The file is likely still access restricted.")
            raise Exception("Google Drive link returned HTML/Error page.")

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        st.info(f"Downloading model (approx. {total_size / (1024*1024):.2f} MB)...")
        progress_bar = st.progress(0)

        downloaded_size = 0
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    progress_bar.progress(min(100, int((downloaded_size / total_size) * 100)))
            
            if downloaded_size == 0:
                raise Exception("Zero-byte file downloaded.")
        
        progress_bar.progress(100)
        st.success("Model downloaded successfully!")

    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise e
    finally:
        progress_bar.empty()


# --- Load the trained model ---
@st.cache_resource
def load_model():
    model_path = MODEL_PATH
    
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at: {model_path}. Attempting raw download from Google Drive...")
        try:
            # *** USING THE SIMPLIFIED RAW DOWNLOAD FUNCTION ***
            download_file_from_google_drive_raw(GDRIVE_FILE_ID, model_path)
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
        st.error("FINAL FAILURE: If this error persists, the file is corrupted. The only solution is to host the model on a different platform (e.g., Hugging Face).")
        return None

model = load_model()

# --- Define image preprocessing function ---
def preprocess_image(img_data):
    img = Image.open(img_data)
    img = img.resize((150, 150)) # Input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale
    return img_array

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a brain MRI image to get a tumor classification prediction.")

if model is None:
    st.warning("Model could not be loaded. Please host the model on a stable platform like Hugging Face if the error persists.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        processed_image = preprocess_image(uploaded_file)

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
            """
            This model is a **Convolutional Neural Network (CNN)** 

[Image of Convolutional Neural Network Architecture]
 trained to classify brain MRI images 

[Image of Brain MRI with Tumor]
 into one of four categories: 
            **Glioma Tumor**, **Meningioma Tumor**, **No Tumor**, or **Pituitary Tumor**.
            """
        )
        st.markdown(
            "**Disclaimer**: This is a demo for educational purposes and should not be used for medical diagnosis."
        )
