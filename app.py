import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests 

# --- Google Drive Configuration ---
GDRIVE_FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
MODEL_PATH = 'brain_tumor_cnn_model.h5' 

# --- Utility Function to Download from Google Drive (Robust version) ---
def download_file_from_google_drive(file_id, destination):
    # ... (Robust download function as provided in the previous response)
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    params = {'id': file_id}
    if token:
        st.info("Bypassing Google Drive large file warning...")
        params['confirm'] = token
        
    response = session.get(URL, params=params, stream=True)
    
    if 'text/html' in response.headers.get('content-type', ''):
        st.error("Download Failed: The response was HTML, not the model file. Check permissions.")
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


# Set Streamlit page config (as before)
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
            # Using safe_mode=False for compatibility
            model = tf.keras.models.load_model(model_path, safe_mode=False) 
        st.success("Model loaded!")
        return model
    except Exception as e:
        # This is the final error you are seeing.
        st.error(f"Error loading model: {e}")
        st.error("FINAL TROUBLESHOOTING: The file downloaded is corrupted. You *must* verify the Google Drive file's sharing setting is set to **'Anyone with the link'**.")
        return None

model = load_model()

# --- Define image preprocessing function ---
def preprocess_image(img_data):
    # ... (Pre-processing function body remains the same)
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
    st.warning("Model could not be loaded. Please correct the Google Drive file permissions and restart the app.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    # ... (Rest of the Streamlit UI code remains the same)
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
            "This model is a Convolutional Neural Network trained to classify brain MRI images into one of four categories: "
            "Glioma Tumor, Meningioma Tumor, No Tumor, or Pituitary Tumor."
        )
        st.markdown(
            "**Disclaimer**: This is a demo for educational purposes and should not be used for medical diagnosis."
        )
