import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model_path = 'efficientnet.h5'
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
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = card_names[predicted_class_index]

    return predicted_class_name, predictions[0][predicted_class_index]

# Streamlit app
st.title('Real-Time Card Image Classification')

# Real-Time Video Stream Detection
run_video = st.checkbox("Start Video Stream")

if run_video:
    cap = cv2.VideoCapture(0)  # Open the default webcam

    if not cap.isOpened():
        st.error("Failed to capture video. Please check your camera settings.")
        st.stop()

    frame_placeholder = st.empty()
    prediction_text = st.empty()

    while run_video:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from the camera.")
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Make prediction
        predicted_class_name, confidence = predict_image(frame_pil)

        # Display the frame with prediction
        frame_placeholder.image(frame_rgb, channels="RGB")
        prediction_text.markdown(f"**Predicted Card: {predicted_class_name}** (Confidence: {confidence:.2f})")

        # Add a break condition to stop the loop
        if not st.checkbox("Continue Video Stream", value=True):
            break

    cap.release()
    frame_placeholder.empty()
