import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import OrderedDict
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
os.environ['YOLO_VERBOSE'] = 'False'
import threading
from readchar import readkey, key
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play, _play_with_ffplay
import io
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
from PIL import Image
import torch
import imagehash
import configparser
import random
import string
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import subprocess
import signal
import time

mtcnn = MTCNN(margin=40, select_largest=False, selection_method='probability', keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.8])
tracedModel = torch.jit.load('tracedModel/inceptionResnetV1Traced.pt')
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

#Setup mediapipe
mpFaceDetection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh

faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.8, model_selection=1)
faceMesh = mpFaceMesh.FaceMesh()
mpDrawing = mp.solutions.drawing_utils
# ====== Logging ======
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ====== AKAZE so sánh ảnh ======
def compareAkaze(image1, image2):
    if image1 is not None and image2 is not None and image1.size > 0 and image2.size > 0:

        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        akaze = cv2.AKAZE_create()

        _, des1 = akaze.detectAndCompute(img1, None)
        _, des2 = akaze.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False

        if des1.dtype != des2.dtype:
            des2 = des2.astype(des1.dtype)

        if des1.shape[1] != des2.shape[1]:
            raise ValueError("Descriptors have different number of columns!")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        totalMatches = len(matches)
        if totalMatches == 0:
            return False

        goodMatches = sum([1 for match in matches if match.distance < 50])
        similarityPercentage = round(goodMatches / totalMatches, 2)
        print(f'similarityPercentage: {similarityPercentage}')
        return similarityPercentage > 0.6
    else:
        return False

# ====== COMPUTE IOU ======

def computeIou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    xA_int = max(xA, xB)
    yA_int = max(yA, yB)
    xB_int = min(xA + wA, xB + wB)
    yB_int = min(yA + hA, yB + hB)

    interArea = max(0, xB_int - xA_int) * max(0, yB_int - yA_int)

    boxAArea = wA * hA
    boxBArea = wB * hB

    unionArea = boxAArea + boxBArea - interArea

    iou = interArea / unionArea
    return iou

# ====== Centroid Tracker ======

class CentroidTracker:
    def __init__(self, maxDisappeared=10, iouThreshold=0.5):
        self.nextObjectId = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.iouThreshold = iouThreshold 

    def register(self, box):
        self.objects[self.nextObjectId] = box
        self.disappeared[self.nextObjectId] = 0
        self.nextObjectId += 1

    def deregister(self, objectId):
        del self.objects[objectId]
        del self.disappeared[objectId]

    def update(self, rects):
        if len(rects) == 0:
            for objectId in list(self.disappeared.keys()):
                self.disappeared[objectId] += 1
                if self.disappeared[objectId] > self.maxDisappeared:
                    self.deregister(objectId)
            return self.objects

        usedRows = set()
        usedCols = set()

        for objectId, trackedBox in list(self.objects.items()):
            bestMatchIdx = -1
            bestIoU = 0
            for i, newBox in enumerate(rects):
                if i in usedCols:
                    continue
                iou = computeIou(trackedBox, newBox)
                if iou > self.iouThreshold and iou > bestIoU:
                    bestIoU = iou
                    bestMatchIdx = i

            if bestMatchIdx != -1:
                self.objects[objectId] = rects[bestMatchIdx]
                self.disappeared[objectId] = 0
                usedCols.add(bestMatchIdx)
                usedRows.add(objectId)

        for i, newBox in enumerate(rects):
            if i not in usedCols:
                self.register(newBox)

        for objectId in list(self.objects.keys()):
            if objectId not in usedRows:
                self.disappeared[objectId] += 1
                if self.disappeared[objectId] > self.maxDisappeared:
                    self.deregister(objectId)
        return self.objects

class Color:
    Red = (0, 0, 255)
    Green = (0, 255, 0)
    Blue = (255, 0, 0)
    Yellow = (0, 255, 255)
    Cyan = (255, 255, 0)
    Magenta = (255, 0, 255)
    White = (255, 255, 255)
    Black = (0, 0, 0)
    Gray = (169, 169, 169)
    Orange = (0, 165, 255)
    Pink = (203, 192, 255)
    Purple = (128, 0, 128)
    Brown = (42, 42, 165)
    Violet = (238, 130, 238)
    Lime = (0, 255, 0)
    Olive = (0, 128, 128)
    Teal = (128, 128, 0)
    Silver = (192, 192, 192)
    Gold = (0, 215, 255)

def checkLight(image, threshold = 90):
    avgBrightness = 0
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avgBrightness = np.mean(grayImage)
    return avgBrightness >= threshold

def faceAngle(nose, leftEye, rightEye, fWidth, fHeight):

    noseX, noseY = int(nose.x * fWidth), int(nose.y * fHeight)
    leftEyeX, leftEyeY = int(leftEye.x * fWidth), int(leftEye.y * fHeight)
    rightEyeX, rightEyeY = int(rightEye.x * fWidth), int(rightEye.y * fHeight)
    
    vector1 = np.array([leftEyeX - noseX, leftEyeY - noseY])

    vector2 = np.array([rightEyeX - noseX, rightEyeY - noseY])

    dotProduct = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosAngle = round(dotProduct / (magnitude1 * magnitude2), 2)
    return cosAngle <= 0.6

def goodFaceAngle(image, width, height):
    goodAngle = False
    if image is not None and image.size > 0:
        results = faceMesh.process(image)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                leftEye = landmarks.landmark[173]
                rightEye = landmarks.landmark[398]
                nose = landmarks.landmark[4]
                goodAngle = faceAngle(nose, leftEye, rightEye, width, height)
    return goodAngle

def checkBlurry(image, threshold=100):
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(grayImage, cv2.CV_64F)
        variance = laplacian.var()
        if variance < threshold:
            return True
    return False

def displayText(frame, x, y, w, light, angle, lowSpeed, blur, color):
    x = x + w
    conditions = [
        (not light, "no light", color.Black),
        (not angle, "look straight", color.Blue),
        (not lowSpeed, "slow down", color.Red),
        (blur, "blur", (167, 80, 167))
    ]
    
    textY = y - 10
    
    for condition, text, rectColor in conditions:
        if condition:
            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            cv2.rectangle(frame, (x, textY - textHeight - 12), (x + textWidth + 10, textY + 5), rectColor, -1)
            
            cv2.putText(frame, text, (x + 4, int(textY - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color.White, 2)
            
            textY += textHeight + 18

def doFaceRecognition(faceImage):
    log("Đang nhận diện khuôn mặt...")
    time.sleep(1)
    names = ["Minh", "Hùng", "Lan", "Thảo", "Dũng"]
    print(f'names: {names}')
    return random.choice(names)

def recognizeFace(faceImage, objectId, trackerRef, trackName):
    originalFace = faceImage.copy()
    tempNames = []

    for i in range(3):
        if objectId in latestFaceImages:
            prevFace = latestFaceImages[objectId]
            if compareAkaze(faceImage, prevFace):
                name = latestNameResults.get(objectId, "Unknown")
                log(f"[ID {objectId}] Ảnh lần {i+1} giống lần trước — dùng lại: {name}")
            else:
                name = doFaceRecognition(faceImage)
                latestFaceImages[objectId] = originalFace
                latestNameResults[objectId] = name
                log(f"[ID {objectId}] Ảnh khác — nhận diện mới: {name}")
        else:
            name = doFaceRecognition(faceImage)
            latestFaceImages[objectId] = originalFace
            latestNameResults[objectId] = name
            log(f"[ID {objectId}] Nhận diện lần đầu: {name}")

        if objectId not in trackerRef.objects:
            log(f"[ID {objectId}] Rời khỏi khung — hủy job.")
            return

        tempNames.append(name)

    if tempNames.count(tempNames[0]) == 3:
        finalName = tempNames[0]
    else:
        finalName = "Unknown"

    if objectId in trackerRef.objects:
        trackName[objectId] = finalName
        log(f"[ID {objectId}] Gán tên: {finalName}")

if __name__ == "__main__":
    # ====== Main ======
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.8, model_selection=1)
    video = "output_video.avi"
    video2 = "output_video2.avi"
    video3 = "6863054282731021257.mp4"
    video4 = "output.mp4"
    video5 = "output1.mp4"
    cap = cv2.VideoCapture(0)

    tracker = CentroidTracker()
    trackName = {}
    recognizedSet = set()
    latestFaceImages = {}
    latestNameResults = {}

    frameCount = 0
    prevTime = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frameCount += 1
        currentTime = time.time()
        if currentTime - prevTime >= 1:
            fps = frameCount
            frameCount = 0
            prevTime = currentTime

        frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(frameRgb)

        rects = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                rects.append((x, y, width, height))

        objects = tracker.update(rects)

        for objectId, (x, y, w, h) in objects.items():
            faceCrop = frame[y:y+h, x:x+w]

            if objectId not in trackName and objectId not in recognizedSet:
                    trackName[objectId] = "Checking..."
                    recognizedSet.add(objectId)
                    thread = threading.Thread(
                        target=recognizeFace,
                        args=(faceCrop, objectId, tracker, trackName)
                    )
                    thread.daemon = True
                    thread.start()

            name = trackName.get(objectId, "New person")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} (ID {objectId})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Tracking + Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

