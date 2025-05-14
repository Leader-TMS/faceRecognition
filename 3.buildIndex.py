import os
import numpy as np
import faiss
import joblib

def buildFaissIndex():
    allEmbeddings = []
    allIds = []

    for npyFile in os.listdir('embeddings'):
        if npyFile.endswith('.npy'):
            personName = os.path.splitext(npyFile)[0]
            emb = np.load(f'embeddings/{npyFile}')
            allEmbeddings.append(emb)
            allIds.extend([personName] * emb.shape[0])

    allEmbeddings = np.vstack(allEmbeddings).astype('float32')
    index = faiss.IndexFlatIP(allEmbeddings.shape[1])
    index.add(allEmbeddings)

    faiss.write_index(index, 'index/faiss.index')
    joblib.dump(allIds, 'index/ids.pkl')
    print(f"[DONE] Xây dựng FAISS index với {len(allIds)} vectors.")

if __name__ == "__main__":
    buildFaissIndex()