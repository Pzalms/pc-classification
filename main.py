import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model_path = 'efficientnet.h5'
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

def predict_image(img):
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = card_names.get(predicted_class_index, 'Unknown')
    return predicted_class_name, predictions[0][predicted_class_index]

def live_prediction():
    st.title('Live Card Image Classification')

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera. Please ensure the camera is connected and accessible.")
        return

    st.write("Click 'Stop Video Stream' to end capturing.")
    stop_button = st.button("Stop Video Stream")

    # Create a placeholder for displaying video
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Convert frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Make predictions
        predicted_class_name, confidence = predict_image(img)

        # Display predictions on frame
        cv2.putText(frame, f'Prediction: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert frame to PIL image and display
        frame_image = Image.fromarray(frame)
        frame_placeholder.image(frame_image, caption='Live Video Feed', use_column_width=True)

        # Break loop if 'Stop' button is pressed
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.write("Video stream ended.")

if __name__ == '__main__':
    live_prediction()
