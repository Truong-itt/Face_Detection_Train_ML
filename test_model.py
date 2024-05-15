import numpy as np 
import os 
from PIL import Image
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from time import sleep


list_name = ['ongtay', 'quanit', 'thaonguyen', 'truong']
# Load the model
models = models.load_model('model-cifar10_10epochs.h5')
face_getector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
# you choose the video
# cam = cv2.VideoCapture("video_example.mp4")

count = 0

while True:
    _, frame = cam.read()
    faces = face_getector.detectMultiScale(frame, 1.3, 5)
    # sleep(0.2)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y:y+h, x:x+w], (100, 100))
        result = np.argmax(models.predict(roi.reshape(-1, 100, 100, 3)))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, list_name[result], (x+15 , y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
    
cam.release()