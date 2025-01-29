# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:36:26 2025

@author: PRAVEEN
"""

from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf

app = Flask(__name__)

# Load model from JSON file
json_file = open('static/modelGG.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create and compile the model
model = model_from_json(loaded_model_json)

# Load weights into the model
model.load_weights('static/model1GG.h5')
print("Loaded model from disk")

def detect_emotion(frame):
    try:
        # Preprocess frame
        resized_frame = cv2.resize(frame, (48, 48))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        input_data = np.expand_dims(gray_frame, axis=(0, -1)) / 255.0

        # Prediction
        predictions = model.predict(input_data)
        emotion = np.argmax(predictions)
        return emotion
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return 0  # Return default emotion index in case of error

def gen_frames():
    try:
        # Try different camera indices
        camera = None
        for index in [0, 1]:  # Try both default and external camera
            print(f"Attempting to open camera at index {index}")
            camera = cv2.VideoCapture(index)
            if camera.isOpened():
                print(f"Successfully opened camera at index {index}")
                break
            else:
                print(f"Failed to open camera at index {index}")
                camera.release()
        
        if camera is None or not camera.isOpened():
            print("Error: Could not open any camera")
            return

        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame")
                break
            else:
                try:
                    # Check frame properties
                    if frame is None or frame.size == 0:
                        print("Error: Invalid frame received")
                        continue

                    emotion = detect_emotion(frame)
                    label = ["Happy", "Sad", "Neutral"][emotion]
                    cv2.putText(frame, f"Emotion: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if not ret:
                        print("Error: Could not encode frame")
                        continue
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error in frame generation: {str(e)}")
                    continue
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
    finally:
        if camera is not None and camera.isOpened():
            camera.release()
            print("Camera released")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('indexz.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
