import cv2
import mediapipe as mp
import time
import numpy as np
import os
from datetime import datetime, timedelta


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
mpDrawing = mp.solutions.drawing_utils

rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"
video = "output_video2.avi"
cap = cv2.VideoCapture(0)
arrayFPS = []
trackers = {}
trackingId = 1

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def compareOrb(image1, image2):
    if not image1.size or not image2.size:
        print('image is empty')
        return 0
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    print(f"shape: {img1.shape} {img2.shape}")
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print(f"Number of keypoints in img1: {len(kp1)}")
    print(f"Number of keypoints in img2: {len(kp2)}")
    if des1 is None or des2 is None:
        saveFaceDirection(img1, "img1")
        saveFaceDirection(img2, "img2")
        print("No descriptors found!")
        return 0

    if des1.dtype != des2.dtype:
        des2 = des2.astype(des1.dtype)

    if des1.shape[1] != des2.shape[1]:
        raise ValueError("Descriptors have different number of columns!")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    totalMatches = len(matches)
    if totalMatches == 0:
        print("No matches found!")
        return 0
    
    goodMatches = sum([1 for match in matches if match.distance < 50])
    similarityPercentage = (goodMatches / totalMatches) * 100

    print(f"compareOrb: {similarityPercentage}")
    return similarityPercentage

def compareAkaze(image1, image2):
    if not image1.size or not image2.size:
        print('Image is empty')
        return 0

    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    print(f"Shape of image1: {img1.shape}, shape of image2: {img2.shape}")

    akaze = cv2.AKAZE_create()

    _, des1 = akaze.detectAndCompute(img1, None)
    _, des2 = akaze.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("No descriptors found!")
        return 0

    if des1.dtype != des2.dtype:
        des2 = des2.astype(des1.dtype)

    if des1.shape[1] != des2.shape[1]:
        raise ValueError("Descriptors have different number of columns!")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    totalMatches = len(matches)
    if totalMatches == 0:
        print("No matches found!")
        return 0

    goodMatches = sum([1 for match in matches if match.distance < 50])
    similarityPercentage = (goodMatches / totalMatches) * 100

    print(f"Similarity percentage: {similarityPercentage}%")
    return similarityPercentage

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
            
            if not trackers:
                trackers[1] = (originalFrame[y:y+h, x:x+w], (x, y, w, h))
                arrFaceTracking.append(1)
            else:
                lenTracker = len(trackers)
                for key, (image, _) in list(trackers.items()):
                    orb = compareAkaze(originalFrame[y:y+h, x:x+w], image)

                    if orb > 50: 
                        trackers[key] = (originalFrame[y:y+h, x:x+w], (x, y, w, h))
                        arrFaceTracking.append(key)
                        break
                    else:
                        arrNotFaceTracking.append(key)
                        newKey = key + 1
                        if newKey not in trackers and newKey > max(trackers.keys()):
                            trackers[newKey] = (originalFrame[y:y+h, x:x+w], (x, y, w, h))
                            arrFaceTracking.append(newKey)

        arrFaceTracking = set(arrFaceTracking)
        arrNotFaceTracking = set(arrNotFaceTracking)
        for num in arrNotFaceTracking:
            if num not in arrFaceTracking:
                del trackers[num]
        for key, (_, xywh) in list(trackers.items()):
            x, y, w, h = xywh

            cv2.putText(frame, f'ID: {key}', (int(x), int(y * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(f"=== END ===")
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
