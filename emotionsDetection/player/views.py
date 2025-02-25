from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import base64
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# initialise Variables 
haarcascade_path = os.path.join(os.getcwd(), 'Train', 'haarcascade_frontalface_default.xml')
emotions_model = os.path.join(os.getcwd(), 'Train', 'model.h5')
emotion_dict = {0:'Anger', 1:'Disgust', 2:'Fear', 3:'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral'}
face_detection = cv2.CascadeClassifier(haarcascade_path)
face_dimension = 48
model.load_weights(emotions_model)

def perform_emotion_detection(image_data):
    # Convert the frame to grayscale
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # No faces detected
        return 'Unknown'

    # Assume the first detected face for simplicity (you may modify based on your requirements)
    (x, y, w, h) = faces[0]

    # Extract the region of interest (ROI) for emotion detection
    roi_gray = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (face_dimension, face_dimension)), -1), 0)

    # Perform emotion detection using your existing model
    prediction = model.predict(cropped_img)
    maxindex = int(np.argmax(prediction))
    detected_emotion = emotion_dict.get(maxindex, 'Unknown')

    return detected_emotion

def music_home(request):
    return render(request, "home.html")

def music_about(request):
    return render(request, "about.html")

def music_player(request):
    return render(request, 'music-player.html')

def detect_emotion(request):
    if request.method == 'POST':
        # Get the JSON data from the frontend
        json_data = json.loads(request.body.decode('utf-8'))
        image_data_base64 = json_data.get('image_data', '')

        # Decode the Base64 string into binary data
        image_data = base64.b64decode(image_data_base64.split(',')[-1])

        # For debugging purposes, save the received image as a file
        with open('received_image.jpg', 'wb') as f:
            f.write(image_data)

        # Perform any further processing (e.g., emotion detection) using the image_data
        try: emotion = perform_emotion_detection(image_data)
        except: emotion = "Not Detected"
        # Return a response back to the frontend
        return JsonResponse({'response': 'Image received successfully!', "emotion":emotion})

    return JsonResponse({'error': 'Invalid request method'})