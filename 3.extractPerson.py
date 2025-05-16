# 3.extractPerson.py
import os
import numpy as np
import torch
from sklearn.preprocessing import normalize
from facenet_pytorch import InceptionResnetV1, MTCNN
import joblib
import cv2

mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

def extractPersonEmbeddings(personName):
    personFolder = os.path.join('dataset', personName)
    embeddings = []

    for imgName in os.listdir(personFolder):
        imgPath = os.path.join(personFolder, imgName)
        img = cv2.imread(imgPath)
        if img is None:
            continue
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(imgRgb)

        if face is not None:
            with torch.no_grad():
                emb = model(face.unsqueeze(0)).numpy()
            embeddings.append(emb.flatten())

    if embeddings:
        embeddings = normalize(np.array(embeddings))
        np.save(f'embeddings/{personName}.npy', embeddings)
        print(f"[DONE] Lưu embedding cho {personName}.")
    else:
        print(f"[WARN] Không tìm thấy khuôn mặt trong ảnh của {personName}.")

if __name__ == "__main__":
    personName = input("Nhập tên người cần trích xuất embedding: ")
    extractPersonEmbeddings(personName)