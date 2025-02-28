import cv2
import mediapipe as mp
import time
import numpy as np
import os
from datetime import datetime, timedelta


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
mpDrawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
arrayFPS = []
trackers = {}

def isInsideRectangle(smallX, smallY, smallW, smallH, largeX, largeY, largeW, largeH):
    if smallX >= largeX and smallY >= largeY:
        if smallX + smallW <= largeX + largeW and smallY + smallH <= largeY + largeH:
            return True
    return False

def detections(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(rgbFrame)

    if results.detections:
        arrFaceTracking = []
        arrNotFaceTracking = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            newW = int(w * 1.75)
            newH = int(h * 1.75)
            newX = int(x - (newW - w) / 2)
            newY = int(y - (newH - h) / 2)

            if not trackers:
                trackers[1] = (newX, newY, newW, newH)
            else:
                for key, (bbox) in list(trackers.items()):
                    largeX, largeY, largeW, largeH = bbox
                    inside = isInsideRectangle(x, y, w, h, largeX, largeY, largeW, largeH)

                    if inside: 
                        trackers[key] = (newX, newY, newW, newH)
                        arrFaceTracking.append(key)
                        break
                    else:
                        arrNotFaceTracking.append(key)
                        newKey = key + 1
                        if newKey not in trackers and newKey > max(trackers.keys()):
                            trackers[newKey] = (newX, newY, newW, newH)
                            arrFaceTracking.append(newKey)

            cv2.rectangle(frame, (newX, newY), (newX + newW, newY + newH), (0, 255, 0), 2)

        arrFaceTracking = set(arrFaceTracking)
        arrNotFaceTracking = set(arrNotFaceTracking)

        for num in arrNotFaceTracking:
            if num not in arrFaceTracking:
                del trackers[num]
        for key, (bbox) in list(trackers.items()):
            x, y, w, h = bbox

            new_w = int(w * 0.8)
            new_h = int(h * 0.8)
            new_x = int(x + (w - new_w) / 2)
            new_y = int(y + (h - new_h) / 2)
            
            cv2.putText(frame, f'ID: {key}', (new_x, int(new_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

if __name__ == "__main__":
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
            arrNotFaceTracking = []
            for index, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                newW = int(w * 1.75)
                newH = int(h * 1.75)
                newX = int(x - (newW - w) / 2)
                newY = int(y - (newH - h) / 2)

                if not trackers:
                    trackers[1] = (newX, newY, newW, newH)
                else:
                    lenTracker = len(trackers)
                    for key, (bbox) in list(trackers.items()):
                        largeX, largeY, largeW, largeH = bbox
                        inside = isInsideRectangle(x, y, w, h, largeX, largeY, largeW, largeH)

                        if inside: 
                            trackers[key] = (newX, newY, newW, newH)
                            arrFaceTracking.append(key)
                            break
                        else:
                            arrNotFaceTracking.append(key)
                            newKey = key + 1
                            if newKey not in trackers and newKey > max(trackers.keys()):
                                trackers[newKey] = (newX, newY, newW, newH)
                                arrFaceTracking.append(newKey)

                # if isInsideRectangle(x, y, w, h, newX, newY, newW, newH):
                #     print("Hình chữ nhật nhỏ nằm hoàn toàn bên trong hình chữ nhật lớn.")
                # else:
                #     print("Hình chữ nhật nhỏ không nằm hoàn toàn bên trong hình chữ nhật lớn.")

                cv2.rectangle(frame, (newX, newY), (newX + newW, newY + newH), (0, 255, 0), 2)

            arrFaceTracking = set(arrFaceTracking)
            arrNotFaceTracking = set(arrNotFaceTracking)

            for num in arrNotFaceTracking:
                if num not in arrFaceTracking:
                    del trackers[num]
            for key, (bbox) in list(trackers.items()):
                x, y, w, h = bbox

                new_w = int(w * 0.8)
                new_h = int(h * 0.8)
                new_x = int(x + (w - new_w) / 2)
                new_y = int(y + (h - new_h) / 2)
                
                cv2.putText(frame, f'ID: {key}', (new_x, int(new_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)

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
