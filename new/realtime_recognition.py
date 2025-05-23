# import cv2
# import faiss
# import joblib
# import numpy as np
# import time
# from insightface.model_zoo import get_model
# from insightface.utils import face_align

# detModel = get_model('models/det_500m.onnx')
# detModel.prepare(ctx_id=-1, input_size=(640, 640))

# recModel = get_model('models/w600k_mbf.onnx')
# recModel.prepare(ctx_id=-1, input_size=(640, 640))

# faissIndex = faiss.read_index('indexWithInsight/faiss.index')
# labelList = joblib.load('indexWithInsight/labels.pkl')

# cap = cv2.VideoCapture(0)

# while True:
#     startTime = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)

#     bboxes, kpss = detModel.detect(frame, max_num=3)

#     for i, box in enumerate(bboxes):
#         x1, y1, x2, y2 = box[:4].astype(int)
#         score = box[4]
#         if score < 0.6:
#             continue

#         faceCrop = frame[y1:y2, x1:x2]

#         if faceCrop.size == 0:
#             continue
#         faceHeight = y2 - y1
#         faceWidth = x2 - x1
#         kps = kpss[i]
#         alignedFace = face_align.norm_crop(frame, landmark=kps)
#         embedding = recModel.get_feat(alignedFace)
#         emb = embedding.reshape(1, -1).astype('float32')
#         faiss.normalize_L2(emb)
#         D, I = faissIndex.search(emb, 1)
#         sim = D[0][0]
#         label = labelList[I[0][0]] if sim > 0.5 else "Unknown"

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{label} ({sim:.2f})", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     endTime = time.time()
#     fps = 1 / (endTime - startTime)
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("FaceRecognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import faiss
# import joblib
# import numpy as np
# import time
# import mediapipe as mp
# from insightface.app import FaceAnalysis

# mpFaceDetection = mp.solutions.face_detection
# faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.2, model_selection=1)

# app = FaceAnalysis(name='buffalo_sc', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=-1)

# index = faiss.read_index('indexWithInsight/faiss.index')
# labels = joblib.load('indexWithInsight/labels.pkl')

# cap = cv2.VideoCapture(0)

# while True:
#     startTime = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = faceDetection.process(frameRgb)

#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             x1 = int(bboxC.xmin * iw)
#             y1 = int(bboxC.ymin * ih)
#             w = int(bboxC.width * iw)
#             h = int(bboxC.height * ih)
#             cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
#             centerX = x1 + w // 2
#             centerY = y1 + h // 2

#             scale = 2
#             cropW = int(w * scale)
#             cropH = int(h * scale)

#             x1Crop = max(0, centerX - cropW // 2)
#             y1Crop = max(0, centerY - cropH // 2)
#             x2Crop = min(iw, x1Crop + cropW)
#             y2Crop = min(ih, y1Crop + cropH)

#             faceCrop = frame[y1Crop:y2Crop, x1Crop:x2Crop]

#             if faceCrop.size > 0:
#                 faces = app.get(faceCrop)
#                 for face in faces:
#                     emb = face.embedding.reshape(1, -1).astype('float32')
#                     faiss.normalize_L2(emb)
#                     D, I = index.search(emb, 1)

#                     sim = D[0][0]
#                     label = labels[I[0][0]] if sim > 0.5 else "Unknown"

#                     cv2.rectangle(frame, (x1Crop, y1Crop), (x2Crop, y2Crop), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{label} ({sim:.2f})", (x1Crop, y1Crop - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     endTime = time.time()
#     fps = 1 / (endTime - startTime)
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("Face Recognition", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# CHECK TIME
import cv2
import faiss
import joblib
import numpy as np
import time
import mediapipe as mp
from insightface.model_zoo import get_model
from insightface.utils import face_align

recModel = get_model('models/w600k_mbf.onnx')
recModel.prepare(ctx_id=-1)

faissIndex = faiss.read_index('indexWithInsight/faiss.index')
labelList = joblib.load('indexWithInsight/labels.pkl')

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while True:
    loopStart = time.time()

    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    t1 = time.time()
    print(f"[Camera Read] {t1 - t0:.4f} sec")

    # frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)
    h, w = frame.shape[:2]
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t2 = time.time()
    print(f"[Resize + BGR->RGB] {t2 - t1:.4f} sec")

    results = faceMesh.process(rgbFrame)
    t3 = time.time()
    print(f"[MediaPipe FaceMesh] {t3 - t2:.4f} sec")

    bboxes = []
    kpss = []

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            lm = faceLandmarks.landmark
            kps = np.array([
                [lm[468].x * w,  lm[468].y * h],    # 0 - Left eye
                [lm[473].x * w, lm[473].y * h],   # 1 - Right eye
                [lm[4].x * w,  lm[4].y * h],    # 2 - Nose
                [lm[61].x * w,  lm[61].y * h],    # 3 - Left mouth
                [lm[291].x * w, lm[291].y * h],   # 4 - Right mouth
            ], dtype=np.float32)
            kpss.append(kps)
            x_coords = kps[:, 0]
            y_coords = kps[:, 1]
            x1, y1 = np.min(x_coords), np.min(y_coords)
            x2, y2 = np.max(x_coords), np.max(y_coords)
            bbox = np.array([x1, y1, x2, y2, 1.0], dtype=np.float32)
            bboxes.append(bbox)

    t4 = time.time()
    print(f"[Keypoint + BBox Extraction] {t4 - t3:.4f} sec")

    bboxes = np.array(bboxes)
    kpss = np.array(kpss)

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        faceHeight = y2 - y1
        faceWidth = x2 - x1
        kps = kpss[i]
        alignStart = time.time()
        alignedFace = face_align.norm_crop(frame, landmark=kps)
        embedding = recModel.get_feat(alignedFace)
        emb = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(emb)
        D, I = faissIndex.search(emb, 1)
        sim = D[0][0]
        label = labelList[I[0][0]] if sim > 0.5 else "Unknown"
        alignEnd = time.time()
        print(f"[Align + Embedding + Search] Face {i+1}: {alignEnd - alignStart:.4f} sec")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({sim:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    loopEnd = time.time()
    totalLoop = loopEnd - loopStart
    fps = 1 / totalLoop if totalLoop > 0 else 0
    print(f"[Total Frame Time] {totalLoop:.4f} sec | FPS: {fps:.2f}\n")

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Recognition with MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
