import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

mtcnn = MTCNN(keep_all=True)
inception_model = InceptionResnetV1(pretrained='vggface2').eval()

dataset_dir = 'dataset'

embeddings = []
labels = []
for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    print(f'Person: {person_name}')
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            print(f'Person: {person_name} - {img_name}')
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = mtcnn(img_rgb)
            
            if faces is not None and len(faces) > 0:
                # lấy embedding cho mỗi khuôn mặt
                for face in faces:
                    embedding = inception_model(face.unsqueeze(0)).detach().numpy()
                    embeddings.append(embedding.flatten())
                    labels.append(person_name)
            else:
                print(f'No face found in {img_name}')
                
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

embeddings = np.array(embeddings)

X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=150)

# huấn luyện mô hình SVM
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# kiểm tra huấn luyện
accuracy = svm.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# lưu mô hình SVM và LabelEncoder
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Huấn luyện xong và mô hình đã được lưu.")
