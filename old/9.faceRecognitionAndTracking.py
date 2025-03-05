import cv2
import mediapipe as mp
import time
import numpy as np
import os
from datetime import datetime, timedelta


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
mpDrawing = mp.solutions.drawing_utils

video = "output_video2.avi"
cap = cv2.VideoCapture(video)
arrayFPS = []

while cap.isOpened():
    start =  time.time()
    ret, frame = cap.read()
    if not ret:
        break

    originalFrame = frame.copy()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(rgbFrame)

    if results.detections:
        arrFaceTracking = []
        for index, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    end = time.time() 
    totalTime = end - start
    fps = int(1 / totalTime)
    if fps not in arrayFPS:
        arrayFPS.append(fps)
    minFPS = min(arrayFPS)
    maxFPS = max(arrayFPS)
    avgFPS = sum(arrayFPS) / len(arrayFPS) if arrayFPS else 0
    text = f'FPS: {fps} maxFPS:{maxFPS} avgFPS:{int(avgFPS)} minFPS:{minFPS}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.8, 2
    textColor, bgColor = (255, 255, 255), (167, 80, 167)

    (textWidth, textHeight), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = 30, 30

    cv2.rectangle(frame, (x - 10, y - textHeight - 10), (x + textWidth + 10, y + 10), bgColor, -1)

    cv2.putText(frame, text, (x, y), font, scale, textColor, thickness)
    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
