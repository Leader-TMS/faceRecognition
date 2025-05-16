import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

faceApp = FaceAnalysis(name='buffalo_s')
faceApp.prepare(ctx_id=0)

datasetPath = "dataset"
embeddingPath = "embeddingsWithInsight"
os.makedirs(embeddingPath, exist_ok=True)

personName = input("Nhap ten thu muc nguoi dung trong 'dataset': ").strip()
personDir = os.path.join(datasetPath, personName)

if not os.path.isdir(personDir):
    print(f"[LOI] Thu muc khong ton tai: {personDir}")
else:
    personEmbeddings = []
    for imageName in os.listdir(personDir):
        imagePath = os.path.join(personDir, imageName)
        image = cv2.imread(imagePath)
        if image is None:
            continue

        detectedFaces = faceApp.get(image)
        if detectedFaces:
            personEmbeddings.append(detectedFaces[0].normed_embedding)

    if personEmbeddings:
        personEmbeddings = np.array(personEmbeddings, dtype=np.float32)
        np.save(os.path.join(embeddingPath, f"{personName}.npy"), personEmbeddings)
        print(f"[OK] {personName}: {len(personEmbeddings)} embeddings da duoc luu.")
    else:
        print(f"[CANH BAO] Khong phat hien khuon mat nao trong thu muc: {personName}")
