import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0  # Import EfficientNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Hàm load ảnh từ thư mục
def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    label_counter = 0

    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_map[label_counter] = subfolder  # Gán nhãn cho mỗi thư mục
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh sang RGB
                images.append(img)
                labels.append(label_counter)
            label_counter += 1
    
    return images, labels, label_map

# Load dữ liệu
images, labels, label_map = load_images_from_folder("dataset")

# Resize và chuẩn hóa dữ liệu đầu vào
images = [cv2.resize(img, (224, 224)) for img in images]  # Resize ảnh về kích thước 224x224
images = np.array(images)
images = images.astype('float32') / 255.0  # Chuẩn hóa ảnh về [0, 1]

labels = np.array(labels)

# Tạo mô hình EfficientNetB0 với Transfer Learning
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đặt các lớp trong base model thành không huấn luyện
base_model.trainable = False

# Xây dựng mô hình
model = models.Sequential([
    base_model,  # Thêm EfficientNetB0 đã huấn luyện trên ImageNet
    layers.GlobalAveragePooling2D(),  # Lớp pooling trung bình toàn cục
    layers.Dense(128, activation='relu'),  # Lớp fully connected với 128 neuron
    layers.Dense(len(label_map), activation='softmax')  # Lớp đầu ra với số lớp tương ứng
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)

# Lưu mô hình
model.save("face_recognition_model_efficientnet.h5")

# Lưu label map
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
