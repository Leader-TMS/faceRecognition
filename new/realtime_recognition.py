# import cv2
# import faiss
# import joblib
# import numpy as np
# import time
# from insightface.app import FaceAnalysis

# app = FaceAnalysis(name='buffalo_sc')
# app.prepare(ctx_id=-1)

# index = faiss.read_index('indexWithInsight/faiss.index')
# labels = joblib.load('indexWithInsight/labels.pkl')

# cap = cv2.VideoCapture(0)

# while True:
#     startTime = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)
#     faces = app.get(frame)

#     for face in faces:
#         emb = face.embedding.reshape(1, -1).astype('float32')
#         faiss.normalize_L2(emb)
#         D, I = index.search(emb, 1)

#         sim = D[0][0]
#         label = labels[I[0][0]] if sim > 0.5 else "Unknown"

#         box = face.bbox.astype(int)
#         cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
#         cv2.putText(frame, f"{label} ({sim:.2f})", (box[0], box[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     endTime = time.time()
#     fps = 1 / (endTime - startTime)
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import faiss
import joblib
import numpy as np
import time
import mediapipe as mp
from insightface.app import FaceAnalysis

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.2, model_selection=1)

app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

index = faiss.read_index('indexWithInsight/faiss.index')
labels = joblib.load('indexWithInsight/labels.pkl')

cap = cv2.VideoCapture(0)

while True:
    startTime = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(frameRgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            centerX = x1 + w // 2
            centerY = y1 + h // 2

            scale = 2
            cropW = int(w * scale)
            cropH = int(h * scale)

            x1Crop = max(0, centerX - cropW // 2)
            y1Crop = max(0, centerY - cropH // 2)
            x2Crop = min(iw, x1Crop + cropW)
            y2Crop = min(ih, y1Crop + cropH)

            faceCrop = frame[y1Crop:y2Crop, x1Crop:x2Crop]

            if faceCrop.size > 0:
                faces = app.get(faceCrop)
                for face in faces:
                    emb = face.embedding.reshape(1, -1).astype('float32')
                    faiss.normalize_L2(emb)
                    D, I = index.search(emb, 1)

                    sim = D[0][0]
                    label = labels[I[0][0]] if sim > 0.5 else "Unknown"

                    cv2.rectangle(frame, (x1Crop, y1Crop), (x2Crop, y2Crop), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({sim:.2f})", (x1Crop, y1Crop - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    endTime = time.time()
    fps = 1 / (endTime - startTime)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
