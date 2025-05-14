import cv2
import numpy as np
import faiss
import joblib
from sklearn.preprocessing import normalize
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
mpDrawing = mp.solutions.drawing_utils

mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

index = faiss.read_index('index/faiss.index')
ids = joblib.load('index/ids.pkl')

cap = cv2.VideoCapture(0)

with mpFaceDetection.FaceDetection(min_detection_confidence=0.2, model_selection=1) as faceDetection:
    prevTime = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        imgRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                faceCrop = frame[y:y+h, x:x+w]
                faceRgb = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2RGB)
                face = mtcnn(faceRgb)

                if face is not None:
                    with torch.no_grad():
                        emb = model(face.unsqueeze(0)).numpy()
                    emb = normalize(emb).astype('float32')
                    D, I = index.search(emb, 1)
                    sim = D[0][0]
                    name = ids[I[0][0]] if sim > 0.7 else "Unknown"
                    cv2.putText(frame, f"{name} ({sim:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("FAISS Face Recognition with MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
