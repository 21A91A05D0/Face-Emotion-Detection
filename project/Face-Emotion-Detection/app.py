import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    json_file = open('static/modelGG.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('static/model1GG.h5')
    return model

def detect_emotion(frame, model):
    try:
        # Preprocess frame
        resized_frame = cv2.resize(frame, (48, 48))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        input_data = np.expand_dims(gray_frame, axis=(0, -1)) / 255.0

        # Prediction
        predictions = model.predict(input_data)
        emotion = np.argmax(predictions)
        return ["Happy", "Sad", "Neutral"][emotion]
    except Exception as e:
        st.error(f"Error in emotion detection: {str(e)}")
        return "Unknown"

def main():
    st.title("Real-time Emotion Detection")
    st.write("Live camera feed with emotion detection")

    model = load_model()

    # Create a camera capture object
    camera = cv2.VideoCapture(0)
    
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()
    
    stop_button = st.button("Stop")

    while not stop_button:
        success, frame = camera.read()
        if success:
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotion
            emotion = detect_emotion(frame, model)
            
            # Add emotion text to frame
            cv2.putText(frame_rgb, f"Emotion: {emotion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB")
        else:
            st.error("Error: Could not access camera")
            break

    camera.release()

if __name__ == "__main__":
    main()