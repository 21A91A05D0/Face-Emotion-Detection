# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:36:26 2025

@author: PRAVEEN
"""

from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import subprocess
import time
import json
import os
import base64
from PIL import Image
from io import BytesIO

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
        return ["Happy", "Sad", "Neutral", "Angry", "Surprise"][emotion]
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return "Unknown"

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the image data from the request
        data = request.json['image']
        # Remove the data URL prefix
        image_data = data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect emotion
        emotion = detect_emotion(frame)
        
        return jsonify({'emotion': emotion})
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'emotion': 'Error processing frame'})

@app.route('/')
def index():
    return render_template('indexz.html')

if __name__ == "__main__":
    try:
        # Start ngrok using subprocess with your specific domain
        ngrok_process = subprocess.Popen(['ngrok', 'http', '--domain=exact-wildcat-wholly.ngrok-free.app', '5000'], 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        
        print(f"\n* Your app is now live at: https://exact-wildcat-wholly.ngrok-free.app")
        print("* Share this URL with others to access your emotion detection app!")
        print("* Press CTRL+C to quit\n")
            
        # Run the Flask application
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Running in local mode only...")
        app.run(debug=False, host='0.0.0.0', port=5000)
    finally:
        # Cleanup ngrok process on exit
        if 'ngrok_process' in locals():
            ngrok_process.terminate()
