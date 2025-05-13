import cv2
import mediapipe as mp
import time
from CentroidTracker import CentroidTracker

tracker = CentroidTracker(maxDisappeared=5, maxDistance=60, useCache=True, cacheLifetime=1.5)
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

frameCount = 0
startTime = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameCount += 1
    currentTime = time.time()
    if currentTime - startTime >= 1.0:
        fps = frameCount
        frameCount = 0
        startTime = currentTime

    height, width = frame.shape[:2]
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(rgbFrame)

    rects = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x1 + w)
            y2 = min(height, y1 + h)
            rects.append((x1, y1, x2, y2))

    objects = tracker.update(rects)

    for objectID, (x1, y1, x2, y2) in objects.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow("Face Tracker", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
