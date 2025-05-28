import json
import cv2
import os
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils import face_align
import faiss
import joblib

faissIndex = faiss.read_index('indexWithInsight/faiss.index')
labelList = joblib.load('indexWithInsight/labels.pkl')
jsonPath = "landmarks.json"
imageDir = "captures"

recModel = get_model('models/w600k_mbf.onnx')
recModel.prepare(ctx_id=-1)

with open(jsonPath, "r") as f:
    data = json.load(f)

for imageName, keypoints in data.items():
    imagePath = os.path.join(imageDir, imageName)

    image = cv2.imread(imagePath)
    if image is None:
        print(f"Không tìm thấy ảnh: {imagePath}")
        continue

    kps = np.array([keypoints[str(i)] for i in range(5)], dtype='float32')

    alignedFace = face_align.norm_crop(image, landmark=kps)

    embedding = recModel.get_feat(alignedFace)
    emb = embedding.reshape(1, -1).astype('float32')

    faiss.normalize_L2(emb)

    distances, indices = faissIndex.search(emb, 1)
    similarity = distances[0][0]
    label = labelList[indices[0][0]] if similarity > 0.5 else "Unknown"

    x, y = int(kps[0][0]), int(kps[0][1])
    # cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(image, f"{label} ({similarity:.2f})", (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
