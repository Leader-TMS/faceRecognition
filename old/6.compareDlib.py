# import cv2
# import dlib
# import os
# import numpy as np

# # Khởi tạo Dlib detector và mô hình nhận diện khuôn mặt
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# # Hàm để lấy đặc trưng khuôn mặt từ ảnh
# def get_face_descriptor(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) == 0:
#         return None

#     landmarks = sp(gray, faces[0])  # Chọn khuôn mặt đầu tiên
#     face_descriptor = facerec.compute_face_descriptor(img, landmarks)
#     return np.array(face_descriptor)

# # Quét qua các thư mục con trong dataset để trích xuất và lưu đặc trưng khuôn mặt
# def extract_face_descriptors(dataset_path):
#     descriptors = {}
    
#     for person_folder in os.listdir(dataset_path):
#         person_path = os.path.join(dataset_path, person_folder)
        
#         if os.path.isdir(person_path):
#             person_descriptors = []
#             for image_name in os.listdir(person_path):
#                 image_path = os.path.join(person_path, image_name)
#                 descriptor = get_face_descriptor(image_path)
                
#                 if descriptor is not None:
#                     person_descriptors.append(descriptor)
            
#             # Lưu đặc trưng khuôn mặt của mỗi người trong dictionary
#             if person_descriptors:
#                 descriptors[person_folder] = np.array(person_descriptors)
#                 print(f"Extracted descriptors for {person_folder}")
    
#     return descriptors

# # Lưu đặc trưng khuôn mặt vào file npy
# def save_descriptors(descriptors, filename="face_descriptors.npy"):
#     np.save(filename, descriptors)

# # Quá trình trích xuất và lưu đặc trưng khuôn mặt
# dataset_path = "dataset"  # Thư mục chứa các ảnh khuôn mặt đã cắt
# descriptors = extract_face_descriptors(dataset_path)
# save_descriptors(descriptors)

# --------------------------------------------

import cv2
import dlib
import numpy as np
from scipy.spatial.distance import cosine

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def loadDescriptors(filename="face_descriptors.npy"):
    return np.load(filename, allow_pickle=True).item()

def getFaceDescriptorFromWebcam(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    landmarks = sp(gray, frame)
    faceDescriptor = facerec.compute_face_descriptor(frame, landmarks)
    return np.array(faceDescriptor)

def compareWithDataset(webcamDescriptor, descriptors):
    minDistance = float('inf')
    identity = None

    for person, personDescriptors in descriptors.items():
        for descriptor in personDescriptors:
            distance = cosine(webcamDescriptor, descriptor)
            if distance < minDistance:
                minDistance = distance
                identity = person

    return identity, minDistance

cap = cv2.VideoCapture(0)
descriptors = loadDescriptors()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    webcamDescriptor = getFaceDescriptorFromWebcam(frame)

    if webcamDescriptor is not None:
        identity, distance = compareWithDataset(webcamDescriptor, descriptors)

        if identity is not None and distance < 0.6:
            cv2.putText(frame, f"Identity: {identity} ({round(1 - distance, 2)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
