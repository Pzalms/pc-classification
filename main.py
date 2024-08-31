import os
import requests
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Function to download the model from Google Drive
def download_model_from_link(url, model_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success('Model downloaded successfully.')
    except requests.RequestException as e:
        st.error(f"Error downloading model: {e}")
        raise

# Define model URL and local path
model_url = 'https://drive.google.com/uc?export=download&id=17KRgfd9uHeSqF_0q627fOX7MeTIryoWY'
model_path = 'efficientnet.h5'

# Download the model file if it does not exist
if not os.path.exists(model_path):
    st.write('Downloading model...')
    download_model_from_link(model_url, model_path)

# Load the pre-trained model
try:
    model = load_model(model_path)
    st.success('Model loaded successfully.')
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define image size expected by the model
img_size = (200, 200)

# Dictionary mapping indices to card names
card_names = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    # ... (rest of the mapping)
}

# Function to predict card name
def predict_image(img):
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = card_names[predicted_class_index]
        st.markdown(f"**Prediction Result:** The card image is most likely: **{predicted_class_name}**")
        st.markdown(f"**Prediction Confidence:** Probability: {predictions[0][predicted_class_index]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Streamlit app
st.title('Card Image Classification')

# Upload image functionality
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=False, width=300)
    if st.button("Predict Upload Image"):
        predict_image(img)

# Webcam capture functionality
camera_image = st.camera_input("Take a picture")
if camera_image is not None:
    img = Image.open(camera_image)
    st.image(img, caption='Captured Image', use_column_width=False, width=300)
    if st.button("Predict Webcam Image"):
        predict_image(img)
