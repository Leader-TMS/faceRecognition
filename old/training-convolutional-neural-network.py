import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json

def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_map[label_counter] = subfolder
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(label_counter)
            label_counter += 1
    
    return images, labels, label_map

images, labels, label_map = load_images_from_folder("dataset")

images = [cv2.resize(img, (224, 224)) for img in images]
images = np.array(images)
images = images.reshape(-1, 224, 224, 1)
images = images.astype('float32') / 255.0

labels = np.array(labels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

model.save("face_recognition_model.h5")

with open("label_map.json", "w") as f:
    json.dump(label_map, f)
