from multiprocessing import Queue, Process
import cv2
import mediapipe as mp
import time
import supervision as sv
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import json
import threading
from readchar import readkey, key
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
from PIL import Image
import torch
from torchvision import transforms
import imagehash

# Setup Yolo and Yolo Face
# model = YOLO("yolo/yolo11m.pt")
# model.verbose = False
# -------------------------------------
modelFace = YOLO("yolo/yolov11s-face.pt")
modelFace.verbose = False

class_colors = sv.ColorPalette.from_hex(['#ffff66'])
tracker = sv.ByteTrack()
boundingBoxAnnotator = sv.BoxAnnotator(thickness=2, color=class_colors)
labelAnnotator = sv.LabelAnnotator(color=class_colors, text_color=sv.Color.from_hex("#000000"))
# mtcnn = MTCNN(keep_all = True, thresholds=[0.5, 0.6, 0.6])
mtcnn = MTCNN(keep_all = True, thresholds=[0.7, 0.8, 0.8])

inceptionModel = InceptionResnetV1(pretrained='vggface2').eval()
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

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

# Setup Data
setTrackerName = {}
seeRFID = False
inputRFID = ""
userAreCheckIn = ""
lock = threading.Lock()
color = Color()
checking = None
targetWFace = 50
checkTimeRecognition = datetime.now()
imageHash = None
trackingIdAssign = None

roi_x1, roi_y1, roi_x2, roi_y2 = 250, 250, 600, 350
points = np.array([[625, 280], [890, 280], [885, 625], [630, 635]], dtype=np.int32)

points = points.reshape((-1, 1, 2))

def textToSpeech(text, speed=1.0):
    def generateAndPlayAudio():
        tts = gTTS(text=text, lang='vi', slow=False)
        fp =io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio = AudioSegment.from_mp3(fp)

        if speed != 1.0:
            audio = audio.speedup(playback_speed=speed)

        play(audio)
        
    thread = threading.Thread(target=generateAndPlayAudio)
    thread.daemon=True
    thread.start()

def updateInfo(data = {}):
    arrUnregistered = []
    arrRegistered = []
    listName = []
    for employeeCode, image in data.items():
        if employeeCode not in data:
            arrUnregistered.append(employeeCode)
        else:
            arrRegistered.append(employeeCode)
            fullName = getEmployeesByCode(employeeCode)["full_name"]
            listName.append(fullName)
            print(f"fullName: {fullName}")
            print(f"image: {image}")
            saveFaceDirection(image, f"evidences/valid/{fullName}")
            saveAttendance(employeeCode)

    listName = ','.join(listName)

    if len(arrRegistered):
        speech = threading.Thread(target=textToSpeech(f"Xin chào {listName}", 1.2))
        speech.daemon = True
        speech.start()
        return False
    elif len(arrUnregistered):
        text = ""
        if len(arrUnregistered) > 1:
            text = f"Có {len(arrUnregistered)} mã nhân viên chưa được đăng ký trong hệ thống."
        else:
            text = "Mã nhân viên chưa được đăng ký trong hệ thống."
        speech = threading.Thread(target=textToSpeech(text, 1.2))
        speech.daemon = True
        speech.start()
        return False

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def faceAuthentication(trackerId, frame):
    try:
        label = "Unknown"
        employeeCode = None
        hFace, wFace = frame.shape[:2]

        if frame is not None and frame.size > 0:
            if wFace < targetWFace:
                wFace = int(wFace * 1.45)
                hFace = int(hFace * 1.45)
                frame = cv2.resize(frame, (wFace, hFace), interpolation=cv2.INTER_LANCZOS4)
            
            if wFace >= targetWFace:
                faceMtcnn = mtcnn(frame)
                if faceMtcnn is not None and len(faceMtcnn) > 0:
                    for i, face in enumerate(faceMtcnn):
                        embedding = inceptionModel(face.unsqueeze(0)).detach().numpy().flatten()
                        embedding = normalize([embedding])
                        labelIndex = svmModel.predict(embedding)[0]
                        prob = svmModel.predict_proba(embedding)[0]
                        probPercent = round(prob[labelIndex], 2)
                        if probPercent >= 0.7:
                            employeeCode = labelEncoder.inverse_transform([labelIndex])[0]
                            employeeCode = getEmployeesByCode(employeeCode)
                            if employeeCode:
                                label = employeeCode['full_name']
                                employeeCode = employeeCode['employee_code']
                        else:
                            saveFaceDirection(frame, "evidences/invalid/Unknown")
                            label = "Unknown"
                            employeeCode = None
    except RuntimeError as e:
            print(f"Warning: {e}. Skipping this face region.")
    return label, employeeCode

def getNameFace(frame = None, trackerId = None):
    if frame is not None and frame.size > 0:
        trackerName, employeeCode = faceAuthentication(trackerId, frame)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
            setTrackerName[trackerId]["employeeCode"] = employeeCode
            setTrackerName[trackerId]["Known"] = setTrackerName[trackerId]["Known"] + 1
        else:
            setTrackerName[trackerId]["name"] = "Unknown"
            setTrackerName[trackerId]["employeeCode"] = None
            setTrackerName[trackerId]["Unknown"] = setTrackerName[trackerId]["Unknown"] + 1

def scanRFID():
    global inputRFID, seeRFID
    while True:
        k = readkey()
        with lock:
            if k not in [key.ENTER, key.BACKSPACE]:
                inputRFID += k
            if k == key.ENTER:
                employee = getEmployeesByRFID(inputRFID)
                if employee:
                    userAreCheckIn = employee['full_name']
                    employeeCode = employee['employee_code']
                    saveAttendance(employeeCode)
                    textToSpeech(f"Xin chào {userAreCheckIn}", 1.2)
                else:
                    textToSpeech("Rờ ép ID chưa được đăng ký", 1.2)
                inputRFID = ""
                seeRFID = True
            if k == key.BACKSPACE:
                inputRFID = inputRFID[:-1]

def drawFaceCoordinate(frame, detection):
    x, y, w, h = map(int, detection[0])
    cv2.putText(frame, f'getNameFace', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return x, y, w, h

def checkTime(savedTime, second = 0.3):
    return round(((datetime.now() - savedTime).total_seconds()), 2) >= second

def findNextLarger(arrTrackerId, trackingIdAssign):
    for i in range(len(arrTrackerId)):
        if arrTrackerId[i] > trackingIdAssign:
            return arrTrackerId[i]
    return arrTrackerId[0]

def delAndFindNextLarger(arrTrackerId, trackingIdAssign):
    if trackingIdAssign in setTrackerName:
        del setTrackerName[trackingIdAssign]
    return findNextLarger(arrTrackerId, trackingIdAssign)


def getImageHash(image):
    pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    imageHash = imagehash.phash(pilImage)
    return imageHash

def videoCapture():
    global checkTimeRecognition, trackingIdAssign, imageHash

    rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"
    video = "output_video.avi"
    video2 = "output_video2.avi"
    cap = cv2.VideoCapture(rtsp)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)
    arrayFPS = []
    while cap.isOpened():
        start =  time.time()
        ret, frame = cap.read()
        if not ret:
            break

        originalFrame = frame.copy()
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        results = modelFace(frame)[0]

        detections = sv.Detections.from_ultralytics(results)
        # detections = annotate_frame(frame, detections, bbox_annotator, label_annotator, class_names_dict)
        detections = detections[detections.confidence >= 0.75]
        detections = detections[detections.class_id == 0]


        # scaleFactor = 0.05  # 5%

        # detections.xyxy[:, 0] *= (1 - scaleFactor)
        # detections.xyxy[:, 1] *= (1 - scaleFactor)
        # detections.xyxy[:, 2] *= (1 + scaleFactor)
        # detections.xyxy[:, 3] *= (1 + scaleFactor)

        x = detections.xyxy[:, 0]
        y = detections.xyxy[:, 1]
        w = detections.xyxy[:, 2]
        h = detections.xyxy[:, 3]

        xCenter = (x + w) / 2
        yCenter = (y + h) / 2

        width = w - x
        height = h - y
        size = np.maximum(width, height)

        detections.xyxy[:, 0] = xCenter - size / 2
        detections.xyxy[:, 1] = yCenter - size / 2
        detections.xyxy[:, 2] = xCenter + size / 2
        detections.xyxy[:, 3] = yCenter + size / 2

        # detections = detections[
        #     (detections.xyxy[:, 2] - detections.xyxy[:, 0] >= targetWFace) &
        #     (detections.xyxy[:, 3] - detections.xyxy[:, 1] >= targetWFace)
        # ]

        detections = tracker.update_with_detections(detections)
        arrDetection = {}
        if len(detections):
            for i, detection in enumerate(detections):
                trackerId = detections.tracker_id[i]
                x, y, w, h = map(int, detection[0])
                topPoint = int((x + w) / 2)
                cv2.circle(frame, (topPoint, h), 3, color.Red, -1)
                checkPointLine = cv2.pointPolygonTest(points, (topPoint, h), False)
                checkPointLine = 1
                if checkPointLine > 0:
                    arrDetection[trackerId] = detection
            if len(arrDetection) and checkTime(checkTimeRecognition, 0.2):
                arrTrackerId = sorted(arrDetection.keys()) if arrDetection is not None else []
                if trackingIdAssign is None:
                    trackingIdAssign = min(arrTrackerId, default=None)
                    
                if trackingIdAssign not in arrTrackerId:
                    trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)

                if trackingIdAssign not in setTrackerName:
                    setTrackerName[trackingIdAssign] = {"name": "Unknown", "employeeCode": None, "Unknown": 0, "Known": 0, "authenticate": "", "timekeeping": False, "timeChecked": 0, "timer": datetime.now(), "time": datetime.now()}
                
                if setTrackerName[trackingIdAssign]["timeChecked"] >= 1:
                    if setTrackerName[trackingIdAssign]["timekeeping"]:
                        trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                    else:
                        if setTrackerName[trackingIdAssign]["name"] != "Unknown":
                            percentage = round((setTrackerName[trackingIdAssign]['Unknown'] / setTrackerName[trackingIdAssign]['Known']) * 100, 2)
                            if percentage <= 25:
                                setTrackerName[trackingIdAssign]["timekeeping"] = True
                                x, y, w, h = drawFaceCoordinate(frame, arrDetection[trackingIdAssign])
                                updateInfo({setTrackerName[trackingIdAssign]['employeeCode']: originalFrame[y:h, x:w]})
                                trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                            else:
                                trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                        else:
                            trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                else:
                    isOther = False
                    x, y, w, h = drawFaceCoordinate(frame, arrDetection[trackingIdAssign])
                    dataHash = getImageHash(originalFrame[y:h, x:w])
                    if imageHash is None:
                        imageHash = dataHash
                        isOther = True
                    else:
                        if imageHash != dataHash:
                            imageHash = dataHash
                            isOther = True
                        elif setTrackerName[trackingIdAssign]["name"] != "Unknown":
                            setTrackerName[trackingIdAssign]["Known"] = setTrackerName[trackingIdAssign]["Known"] + 1
                        else:
                            setTrackerName[trackingIdAssign]["Unknown"] = setTrackerName[trackingIdAssign]["Unknown"] + 1
                    
                    if isOther:
                        getNameFace(originalFrame[y:h, x:w], trackingIdAssign)
                    setTrackerName[trackingIdAssign]["timeChecked"] += 0.2

                checkTimeRecognition = datetime.now()

            labels = [
                f"# {tracker_id} {setTrackerName[tracker_id]['name']} {confidence:0.2f} {xyxy[2] - xyxy[0]}px" if tracker_id in setTrackerName else f"# {tracker_id} Unknown {confidence:0.2f} {xyxy[2] - xyxy[0]}px"
                for confidence, tracker_id, xyxy
                in zip(detections.confidence, detections.tracker_id, detections.xyxy)
            ]
            annotatedFrame = boundingBoxAnnotator.annotate(scene=frame, detections=detections)
            annotatedFrame = labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
        else:
            annotatedFrame = frame
        end = time.time() 
        totalTime = end - start

        fps = int(1 / totalTime)
        if fps not in arrayFPS:
            arrayFPS.append(fps)
        minFPS = min(arrayFPS)
        maxFPS = max(arrayFPS)
        avgFPS = sum(arrayFPS) / len(arrayFPS) if arrayFPS else 0

        text = f'FPS: {fps} maxFPS:{maxFPS} avgFPS:{int(avgFPS)} minFPS:{minFPS} trackingId: {trackingIdAssign}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.8, 2
        textColor, bgColor = color.White, (167, 80, 167)

        (textWidth, textHeight), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 30, 30

        cv2.rectangle(annotatedFrame, (x - 10, y - textHeight - 10), (x + textWidth + 10, y + 10), bgColor, -1)

        cv2.putText(annotatedFrame, text, (x, y), font, scale, textColor, thickness)

        # cv2.rectangle(annotatedFrame, (10, 10), (500, 50), (128, 0, 128), -1)

        # cv2.putText(annotatedFrame, f'FPS: {fps} maxFPS:{maxFPS} avgFPS:{int(avgFPS)} minFPS:{minFPS} trackingId: {trackingIdAssign}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color.Olive, 2)
        
        cv2.imshow("ByteTrack", annotatedFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    scanRFIDThread = threading.Thread(target=scanRFID)
    scanRFIDThread.daemon = True
    scanRFIDThread.start()

    videoCaptureThread = threading.Thread(target=videoCapture)
    videoCaptureThread.start()
    videoCaptureThread.join()

    print("All processes finished.")