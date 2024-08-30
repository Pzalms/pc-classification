import os
import requests
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile

# Function to download the model from a direct download link
def download_model_from_link(url, model_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.text('Model downloaded successfully.')
    except requests.RequestException as e:
        st.error(f"Error downloading model: {e}")
        raise

# Define model URL and local path
model_url = 'https://drive.usercontent.google.com/download?id=17KRgfd9uHeSqF_0q627fOX7MeTIryoWY&export=download&authuser=0&confirm=t&uuid=1f97efb3-b4a1-43ba-907f-b31dc5cd46a4&at=AO7h07cc6j_JasHHvb1t5mcR3eNJ:1725018472593'
model_path = 'efficientnet.h5'

# Download the model file if it does not exist
if not os.path.exists(model_path):
    st.text('Downloading model from direct link...')
    download_model_from_link(model_url, model_path)

# Load the pre-trained model
try:
    model = load_model(model_path)
    st.text('Model loaded successfully.')
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise

# Define image size expected by the model
img_size = (200, 200)

# Dictionary mapping indices to card names
card_names = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades',
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades',
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades',
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades',
    20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades',
    25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades',
    29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades',
    33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades',
    37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades',
    41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades',
    45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades',
    49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades'
}

# Function to process and predict image
def predict_image(img):
    try:
        img = img.convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = card_names[predicted_class_index]

        st.markdown(f"""
        **Prediction Result:**

        The card image is most likely: **{predicted_class_name}**

        **Prediction Confidence:**

        Probability: {predictions[0][predicted_class_index]:.2f}
        """)
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
