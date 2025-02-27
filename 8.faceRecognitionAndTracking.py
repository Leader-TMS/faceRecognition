import cv2
import mediapipe as mp
import time
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
mpDrawing = mp.solutions.drawing_utils

trackers = {}
rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"
video = "output_video2.avi"
# cap = cv2.VideoCapture(2, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(0)
arrayFPS = []
# trackers = cv2.MultiTracker_create()
test = 0
trackingId = 1

# def compareSift(frame1, frame2):
#     img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#     sift = cv2.SIFT_create()

#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(des1, des2)

#     matches = sorted(matches, key=lambda x: x.distance)

#     totalMatches = len(matches)
#     if totalMatches == 0:
#         return 0.0
#     goodMatches = sum([1 for match in matches if match.distance < 30])
#     similarityPercentage = (goodMatches / totalMatches) * 100
#     print(f"compareSift: {similarityPercentage}")

#     return similarityPercentage


def compareOrb(frame1, frame2):
    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    totalMatches = len(matches)
    if totalMatches == 0:
        return 0.0
    goodMatches = sum([1 for match in matches if match.distance < 50])
    similarityPercentage = (goodMatches / totalMatches) * 100
    print(f"compareOrb: {similarityPercentage}")
    return similarityPercentage

def saveFaceDirection(image, folderName, trackingId):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        target_file_name = os.path.join(folderName, f'tracking_id_{trackingId}.jpg')
        cv2.imwrite(target_file_name, image)

while cap.isOpened():
    start =  time.time()
    ret, frame = cap.read()
    if not ret:
        break

    originalFrame = frame.copy()
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(rgbFrame)

    # (success, boxes) = trackers.update(frame)
    # print(f"success: {success} - boxes: {boxes}")
    # for box in boxes:
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)

    if results.detections:
        arrFaceTracking = []
        for index, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            if not trackers:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(originalFrame, (x, y, w, h))
                trackers[trackingId] = (tracker, originalFrame[y:y+h, x:x+w], (x, y, w, h))
                trackingId += 1
            else:
                for key, (tracker, image, _) in list(trackers.items()):
                    orb = compareOrb(originalFrame[y:y+h, x:x+w], image)
                    if orb >= 30 and orb < 50:
                        del trackers[key]
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(originalFrame, (x, y, w, h))
                        trackers[key] = (tracker, originalFrame[y:y+h, x:x+w], (x, y, w, h))
                        break
                    elif orb < 30:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(originalFrame, (x, y, w, h))
                        trackers[trackingId] = (tracker, originalFrame[y:y+h, x:x+w], (x, y, w, h))
                        trackingId += 1
                        break

            # for faceId, (tracker, image, _) in trackers.items():
                # ret, bboxTracker = tracker.update(frame)
                # print()
                # print(f"faceId: {faceId} - ret: {ret} - bboxTracker: {bboxTracker}")
                # if ret:
                #     p1 = (int(bboxTracker[0]), int(bboxTracker[1]))
                #     p2 = (int(bboxTracker[0] + bboxTracker[2]), int(bboxTracker[1] + bboxTracker[3]))
                #     cv2.rectangle(frame, p1, p2, (255, 255, 0), 2)

                # orb = compareOrb(originalFrame[y:y+h, x:x+w], image)
                # if orb >= 30 and orb < 50:
                #     del trackers[faceId]
                #     tracker = cv2.TrackerCSRT_create()
                #     tracker.init(originalFrame, (x, y, w, h))
                #     trackers[faceIds] = (tracker, rgbFrame[y:y+h, x:x+w], (x, y, w, h))
                #     break
                # elif orb < 30:
                #     tracker = cv2.TrackerCSRT_create()
                #     tracker.init(originalFrame, (x, y, w, h))
                #     trackers[faceIds] = (tracker, rgbFrame[y:y+h, x:x+w], (x, y, w, h))
                #     faceIds += 1
                #     break

            # print(f"trackers: {trackers if trackers else 'None'}")
            # if test == 0:
            #     tracker = cv2.TrackerCSRT_create()
            #     tracker.init(originalFrame, (x, y, w, h))
            #     trackers[faceIds] = (tracker, rgbFrame[y:y+h, x:x+w], (x, y, w, h))
            #     if faceIds not in arrFaceTracking:
            #         arrFaceTracking.append(faceIds)
            #     faceIds += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        test += 1
        print(f"=== END ===")

    for key, (tracker, _, _) in trackers.items():
        ret, bbox = tracker.update(frame)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            cv2.putText(frame, f'trackingId: {key}', (int(bbox[0]), int(bbox[1] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1.5)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            print(f"Tracker for face {key} lost!")
            del trackers[key]

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
