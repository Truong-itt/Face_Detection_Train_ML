import numpy as np 
import os 
from PIL import Image
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from time import sleep


TRAIN_DATA = "datasets/train-data"
TEST_DATA = "datasets/test-data"
Xtrain = []
Xtest = []

dict_xtrain = {'ongtay': [1, 0, 0, 0], 'quanit': [0, 1, 0, 0], 'thaonguyen': [0, 0, 1, 0], 'truong': [0, 0, 0, 1]}
dict_xtest = {'ongtay_test': [1, 0, 0, 0], 'quanit_test': [0, 1, 0, 0], 'thaonguyen_test': [0, 0, 1, 0], 'truong_test': [0, 0, 0, 1]}
def Extract(dirfile, lst_data, dict):
    for whatever in os.listdir(dirfile):

        whatever_path = os.path.join(dirfile, whatever)
        print(whatever_path)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):

            filename_path = os.path.join(whatever_path, filename)
            
            label = filename_path.split('\\')[1]
            
            img = np.array(Image.open(filename_path))
            lst_filename_path.append((img, dict[label]))
        lst_data.extend(lst_filename_path)
    return lst_data


Xtrain = Extract(TRAIN_DATA, Xtrain, dict_xtrain)
Xtest = Extract(TEST_DATA, Xtest, dict_xtest)

# dam bao tinh cong bang du lieu
np.random.shuffle(Xtrain)
np.random.shuffle(Xtrain)



# Building the model
model_training_first = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.15),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Print model summary
model_training_first.summary()
# Compile the model
# Train the model
# Save the model

model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

model_training_first.fit(np.array([x[0] for i, x in enumerate(Xtrain)]), np.array([y[1] for i, y in enumerate(Xtrain)]), epochs=10)

model_training_first.save('model-cifar10_10epochs.h5')

