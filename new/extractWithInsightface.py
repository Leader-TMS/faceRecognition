import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Khởi tạo model insightface
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0)  # Dùng CPU, nếu có GPU bạn có thể để ctx_id=0

DATASET = "dataset"
EMBED_DIR = "embeddingsWithInsight"
os.makedirs(EMBED_DIR, exist_ok=True)

for person in os.listdir(DATASET):
    person_dir = os.path.join(DATASET, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if faces:
            embeddings.append(faces[0].normed_embedding)

    if embeddings:
        embeddings = np.array(embeddings, dtype=np.float32)  # Không cần normalize lại
        np.save(os.path.join(EMBED_DIR, f"{person}.npy"), embeddings)
        print(f"[OK] {person}: {len(embeddings)} embeddings")
    else:
        print(f"[WARN] Không phát hiện được khuôn mặt trong {person}")
