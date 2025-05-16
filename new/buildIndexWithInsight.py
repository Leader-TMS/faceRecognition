# build_index.py
import os
import numpy as np
import faiss
import joblib

EMBED_DIR = 'embeddingsWithInsight'
INDEX_DIR = 'indexWithInsight'
os.makedirs(INDEX_DIR, exist_ok=True)

all_embs = []
labels = []

for file in os.listdir(EMBED_DIR):
    if not file.endswith(".npy"):
        continue
    name = os.path.splitext(file)[0]
    embs = np.load(os.path.join(EMBED_DIR, file))
    all_embs.append(embs)
    labels.extend([name] * len(embs))

all_embs = np.vstack(all_embs).astype('float32')
faiss.normalize_L2(all_embs)

index = faiss.IndexFlatIP(all_embs.shape[1])
index.add(all_embs)

faiss.write_index(index, os.path.join(INDEX_DIR, 'faiss.index'))
joblib.dump(labels, os.path.join(INDEX_DIR, 'labels.pkl'))

print(f"[DONE] Đã xây index với {len(labels)} vector.")
