import os
import numpy as np
import cv2
import torch
from sklearn.preprocessing import normalize
from facenet_pytorch import InceptionResnetV1, MTCNN

mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

datasetDir = "dataset"
embeddingDir = "embeddings"

os.makedirs(embeddingDir, exist_ok=True)

def extractEmbeddingsFor(personName):
    folder = os.path.join(datasetDir, personName)
    embeddings = []

    for imgName in os.listdir(folder):
        imgPath = os.path.join(folder, imgName)
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
        np.save(os.path.join(embeddingDir, f"{personName}.npy"), embeddings)
        print(f"[OK] Đã trích {len(embeddings)} embedding cho: {personName}")
    else:
        print(f"[WARN] Không có khuôn mặt hợp lệ trong: {personName}")

def main():
    existingEmbeddings = {
        os.path.splitext(f)[0]
        for f in os.listdir(embeddingDir)
        if f.endswith(".npy")
    }

    for personName in os.listdir(datasetDir):
        personPath = os.path.join(datasetDir, personName)
        if not os.path.isdir(personPath):
            continue
        if personName not in existingEmbeddings:
            extractEmbeddingsFor(personName)
        else:
            print(f"[SKIP] {personName} đã có embedding.")

if __name__ == "__main__":
    main()