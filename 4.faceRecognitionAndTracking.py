from multiprocessing import Queue, Process
import cv2
import mediapipe as mp
import time
import supervision as sv
from facenet_pytorch import MTCNN, InceptionResnetV1
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

# Setup Yolo and Yolo Face
model = YOLO("yolo/yolo11m.pt")
model.verbose = False
tracker = sv.ByteTrack()
boundingBoxAnnotator = sv.BoxAnnotator()
labelAnnotator = sv.LabelAnnotator()
# -------------------------------------
modelFace = YOLO("yolo/yolov11s-face.pt")
modelFace.verbose = False
mtcnn = MTCNN(keep_all = True, thresholds=[0.5, 0.6, 0.6])
inception_model = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load('svmModel.pkl')
label_encoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"
video = "videos/6114574701101869790.mp4"
cap = cv2.VideoCapture(rtsp)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
fps = cap.get(cv2.CAP_PROP_FPS)

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
targetWFace = 60
checkTimeRecognition = datetime.now()
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
    # try:
        # with open(file_name, 'r', encoding='utf-8') as json_file:
        #     data = json.load(json_file)
    # except FileNotFoundError:
    #     speech = threading.Thread(target=textToSpeech("Không tìm thấy tập tin!", 1.2))
    #     speech.daemon = True
    #     speech.start()
    #     return False
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
            # data[employeeCode]['total_scan'] += 1
            # current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            # data[employeeCode]['scanning_time'].append(current_time)
            # with open(file_name, 'w', encoding='utf-8') as json_file:
            #     json.dump(data, json_file, ensure_ascii=False, indent=4)

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
    # return data[employeeCode]['name']

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def saveEvidence(result):
    global inputRFID, seeRFID
    nameInRFID = updateInfo("timekeepingData.json", result)
    # if not nameInRFID:
    #     setTrackerName[trackerId]["authenticate"] = "Invalid RFID info"
    #     saveFaceDirection(frame, "evidences/invalid/" + setTrackerName[trackerId]["name"])
    # else:
    #     setTrackerName[trackerId]["name"] = nameInRFID
    #     setTrackerName[trackerId]["authenticate"] = "Valid RFID info"
    #     saveFaceDirection(frame, "evidences/valid/" + setTrackerName[trackerId]["name"])
    # seeRFID = False
    # inputRFID = ""
    # return nameInRFID

def faceRecognition(frame):
    try:
        label = "Unknown"
        wFace = 0
        maxWFace = 0
        maxHFace = 0
        onlyFace = None
        employeeCode = None

        results = modelFace(frame)
        for i, box in enumerate(results[0].boxes.data):
            x, y, w, h = map(int, box[:4])
            faces = frame[y - 10:h + 10, x - 10:w + 10]
            hFace , wFace = faces.shape[:2]
            if wFace > maxWFace:
                maxWFace = wFace
                maxHFace = hFace
                onlyFace = faces

        if onlyFace is not None and onlyFace.size > 0:
            if maxWFace < targetWFace:
                maxWFace = int(maxWFace * 1.45)
                maxHFace = int(maxHFace * 1.45)
                onlyFace = cv2.resize(onlyFace, (maxWFace, maxHFace), interpolation=cv2.INTER_LANCZOS4)
            
            if maxWFace >= targetWFace:
                faces_mtcnn = mtcnn(onlyFace)
                if faces_mtcnn is not None:
                    for i, face in enumerate(faces_mtcnn):
                        embedding = inception_model(face.unsqueeze(0)).detach().numpy().flatten()
                        embedding = np.array([embedding])
                        label_index = svm_model.predict(embedding)[0]
                        prob = svm_model.predict_proba(embedding)[0]
                        prob_percent = prob[label_index] * 100
                        if prob_percent >= 85:
                            # with open(file_name, 'r', encoding='utf-8') as jsonFile:
                            #     data = json.load(jsonFile)
                            employeeCode = label_encoder.inverse_transform([label_index])[0]
                            employeeCode = getEmployeesByCode(employeeCode)
                            if employeeCode:
                                label = employeeCode['full_name']
                                employeeCode = employeeCode['employee_code']
    except RuntimeError as e:
            print(f"Warning: {e}. Skipping this face region.")
    return label, employeeCode

def getNameFace(frame = None, trackerId = None):
    global inputRFID, seeRFID, userAreCheckIn
    # if trackerId is not None and trackerId not in setTrackerName:
    #     setTrackerName[trackerId] = {"name": "Unknown", "employeeCode": None, "Unknown": 0, "Known": 0, "authenticate": "", "timekeeping": False, "skip": False, "timer": datetime.now(), "time": datetime.now()}
    #     if frame is None:
    #         return 0

    if frame is not None and frame.size > 0:
        # để lại check trường hợp
        # if setTrackerName[trackerId]["name"] == "Unknown":
        #     trackerName, employeeCode = faceRecognition(frame)
        #     if isinstance(trackerName, str) and trackerName != "Unknown":
        #         setTrackerName[trackerId]["name"] = trackerName
        #         setTrackerName[trackerId]["employeeCode"] = employeeCode
        #     setTrackerName[trackerId]["time"] = datetime.now()
        # else:
        trackerName, employeeCode = faceRecognition(frame)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
            setTrackerName[trackerId]["employeeCode"] = employeeCode
            setTrackerName[trackerId]["Known"] = setTrackerName[trackerId]["Known"] + 1
        else:
            setTrackerName[trackerId]["name"] = "Unknown"
            setTrackerName[trackerId]["employeeCode"] = None
            setTrackerName[trackerId]["Unknown"] = setTrackerName[trackerId]["Unknown"] + 1 

        # print(f"setTrackerName: {trackerId, setTrackerName[trackerId]}") 

        # elif seeRFID:
        #     trackerName = saveEvidence(frame, trackerId)
        #     if isinstance(trackerName, str):
        #         setTrackerName[trackerId]["name"] = trackerName

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

def videoCapture():
    global checkTimeRecognition, trackingIdAssign
    while cap.isOpened():
        start =  time.time()
        ret, frame = cap.read()
        if not ret:
            break

        originalFrame = frame.copy()
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence >= 0.7]
        detections = detections[detections.class_id == 0]
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

            print(f"start: {datetime.now()}")

            if len(arrDetection) and checkTime(checkTimeRecognition):
                text = ""
                arrTrackerId = sorted(arrDetection.keys()) if arrDetection is not None else []
                if trackingIdAssign is None:
                    text = "trackingIdAssign is None"
                    trackingIdAssign = min(arrTrackerId, default=None)
                    
                if trackingIdAssign not in arrTrackerId:
                    text = "trackingIdAssign not in arrTrackerId"
                    trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)

                if trackingIdAssign not in setTrackerName:
                    setTrackerName[trackingIdAssign] = {"name": "Unknown", "employeeCode": None, "Unknown": 0, "Known": 0, "authenticate": "", "timekeeping": False, "timeChecked": 0, "timer": datetime.now(), "time": datetime.now()}
                
                if setTrackerName[trackingIdAssign]["timeChecked"] >= 3:
                    text = "timeChecked >= 3"
                    if setTrackerName[trackingIdAssign]["timekeeping"]:
                        text = "timekeeping is True"
                        trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                    else:
                        text = "timekeeping is False"
                        if setTrackerName[trackingIdAssign]["name"] != "Unknown":
                            text = "name is not Unknown"
                            percentage = round((setTrackerName[trackingIdAssign]['Unknown'] / setTrackerName[trackingIdAssign]['Known']) * 100, 2)
                            if percentage < 20:
                                text = "percentage < 20"
                                setTrackerName[trackingIdAssign]["timekeeping"] = True
                                x, y, w, h = drawFaceCoordinate(frame, arrDetection[trackingIdAssign])
                                updateInfo({setTrackerName[trackingIdAssign]['employeeCode']: originalFrame[y:h, x:w]})
                                trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                            else:
                                text = "percentage >= 20"
                                trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                        else:
                            text = "name is Unknown"
                            trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                else:
                    text = "timeChecked < 3"
                    x, y, w, h = drawFaceCoordinate(frame, arrDetection[trackingIdAssign])
                    getNameFace(originalFrame[y:h, x:w], trackingIdAssign)
                    setTrackerName[trackingIdAssign]["timeChecked"] += 0.3

                checkTimeRecognition = datetime.now()

                print(f"{text}: {arrTrackerId} - {trackingIdAssign}")
            
            print(f"end: {datetime.now()}")
            labels = [
                f"# {tracker_id} {setTrackerName[tracker_id]['name']} {confidence:0.2f}" if tracker_id in setTrackerName else f"# {tracker_id} Unknown {confidence:0.2f}"
                for confidence, tracker_id
                in zip(detections.confidence, detections.tracker_id)
            ]
            annotatedFrame = boundingBoxAnnotator.annotate(scene=frame, detections=detections)
            annotatedFrame = labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
        else:
            annotatedFrame = frame
        end = time.time() 
        totalTime = end - start

        fps = 1 / totalTime
        cv2.putText(annotatedFrame, f'FPS: {int(fps)} inputRFID:{inputRFID} trackingIdAssign: {trackingIdAssign}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)
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