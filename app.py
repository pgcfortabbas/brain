import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# 1. Define class labels and target size
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
target_size = (150, 150)

# 2. Streamlit app title
st.title('Brain Tumor Classification App')

# 3. File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define the Google Drive file ID and local path
# Extracted from URL: https://drive.google.com/file/d/1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE/view?usp=drive_link
FILE_ID = '1T3iiKWXTgMBW1zz26x88JVp6s2o1kAdE'
MODEL_PATH = 'brain_tumor_classifier.h5'
DRIVE_URL = f'https://drive.google.com/uc?id={FILE_ID}'

# 4. Load Model with Caching
@st.cache_resource
def load_model_from_drive():
    """
    Downloads the model from Google Drive if not present, 
    and returns the loaded Keras model.
    """
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... this may take a moment.")
        try:
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            st.success("Download complete!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model_from_drive()

if model is not None:
    st.success("Model loaded successfully!")
else:
    st.warning("Model could not be loaded. Please check the Drive link or internet connection.")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Image preprocessing
    image = Image.open(uploaded_file)
    
    # Ensure image is RGB (handle PNGs with transparency or Grayscale)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = image_array / 255.0 # Rescale pixel values
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

    if st.button('Classify Image') and model is not None:
        # Make prediction
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        predicted_label = class_labels[predicted_class_index]

        st.write(f"Prediction: **{predicted_label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
    elif st.button('Classify Image') and model is None:
        st.warning("Model is not loaded. Cannot classify.")
