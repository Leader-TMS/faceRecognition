import os
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

if __name__ == "__main__":

    mtcnn = MTCNN(keep_all=True)
    inceptionModel = InceptionResnetV1(pretrained='vggface2').eval()

    datasetDir = 'dataset'

    embeddings = []
    labels = []
    for personName in os.listdir(datasetDir):
        personFolder = os.path.join(datasetDir, personName)
        print(f'Person: {personName}')
        if os.path.isdir(personFolder):
            for imgName in os.listdir(personFolder):
                imgPath = os.path.join(personFolder, imgName)
                print(f'Person: {personName} - {imgName}')
                img = cv2.imread(imgPath)
                imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgRgb = cv2.resize(imgRgb, (160, 160), interpolation=cv2.INTER_LANCZOS4)  # Resize để tăng độ chính xác

                faces = mtcnn(imgRgb)
                
                if faces is not None and len(faces) > 0:
                    # lấy embedding cho mỗi khuôn mặt
                    for face in faces:
                        embedding = inceptionModel(face.unsqueeze(0)).detach().numpy()
                        embeddings.append(embedding.flatten())
                        labels.append(personName)
                else:
                    print(f'No face found in {imgName}')
                    
    labelEncoder = LabelEncoder()
    labelsEncoded = labelEncoder.fit_transform(labels)

    # embeddings = np.array(embeddings)

    # Chuẩn hóa embedding
    embeddings = normalize(np.array(embeddings))

    XTrain, XTest, yTrain, yTest = train_test_split(embeddings, labelsEncoded, test_size=0.2, random_state=150)

    # huấn luyện mô hình SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(XTrain, yTrain)

    # kiểm tra huấn luyện
    accuracy = svm.score(XTest, yTest)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # lưu mô hình SVM và LabelEncoder
    joblib.dump(svm, 'svmModel.pkl')
    joblib.dump(labelEncoder, 'labelEncoder.pkl')

    print("Huấn luyện xong và mô hình đã được lưu.")