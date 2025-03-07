
import cv2
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
import joblib
import numpy as np
from datetime import datetime
import os
os.environ['YOLO_VERBOSE'] = 'False'
import threading
from readchar import readkey, key
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
from PIL import Image
import torch
import imagehash
import configparser
import random
import string
import sys

mtcnn = MTCNN(keep_all = True, thresholds=[0.7, 0.8, 0.8])
inceptionModel = InceptionResnetV1(pretrained='vggface2').eval()
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

#Setup mediapipe
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.8, model_selection=0)
mpDrawing = mp.solutions.drawing_utils

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
targetWFace = 50
checkTimeRecognition = datetime.now()
imageHash = None
trackingIdAssign = None
trackers = {}
config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingMyCamera']["USER"]
password = config['settingMyCamera']["PASSWORD"]
ip = config['settingMyCamera']["IP"]
port = config['settingMyCamera']["PORT"]
reading= False
roi_x1, roi_y1, roi_x2, roi_y2 = 250, 250, 600, 350
points = np.array([[625, 280], [890, 280], [885, 625], [630, 635]], dtype=np.int32)

points = points.reshape((-1, 1, 2))

def textToSpeech(text, speed=1.0):
    def generateAndPlayAudio():
        global reading
        reading = True
        tts = gTTS(text=text, lang='vi', slow=False)
        fp =io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio = AudioSegment.from_mp3(fp)

        if speed != 1.0:
            audio = audio.speedup(playback_speed=speed)

        play(audio)
        reading = False

    if reading == False:
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
            saveFaceDirection(image, f"evidences/valid/{fullName}")
            saveData = saveAttendance("face", employeeCode, genUniqueId())
            if saveData == False:
                textToSpeech("Lỗi lưu dữ liệu sever, Vui lòng liên hệ Admin", 1.2)

    listName = ','.join(listName)

    if len(arrRegistered):
        textToSpeech(f"Xin chào {listName}", 1.2)

    elif len(arrUnregistered):
        text = ""
        if len(arrUnregistered) > 1:
            text = f"Có {len(arrUnregistered)} mã nhân viên chưa được đăng ký trong hệ thống."
        else:
            text = "Mã nhân viên chưa được đăng ký trong hệ thống."
        textToSpeech(text, 1.2)

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def faceAuthentication(frame):
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
                        if probPercent >= 0.75:
                            employeeCode = labelEncoder.inverse_transform([labelIndex])[0]
                            employeeCode = getEmployeesByCode(employeeCode)
                            if employeeCode:
                                label = employeeCode['full_name']
                                employeeCode = employeeCode['employee_code']
                        else:
                            saveFaceDirection(frame, "evidences/invalid")
                            label = "Unknown"
                            employeeCode = None
    except RuntimeError as e:
            print(f"Warning: {e}. Skipping this face region.")
    finally:
        return label, employeeCode

def getNameFace(frame = None, trackerId = None):
    if frame is not None and frame.size > 0:
        trackerName, employeeCode = faceAuthentication(frame)
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
                    saveData = saveAttendance("rfid", employeeCode, genUniqueId())
                    if saveData == False:
                        textToSpeech("Lỗi lưu dữ liệu sever, Vui lòng liên hệ Admin", 1.2)
                    else:
                        textToSpeech(f"Xin chào {userAreCheckIn}", 1.2)
                else:
                    textToSpeech("Rờ ép ID chưa được đăng ký", 1.2)
                inputRFID = ""
                seeRFID = True
            if k == key.BACKSPACE:
                inputRFID = inputRFID[:-1]

def drawFaceCoordinate(frame, detection):
    x, y, w, h = detection
    cv2.putText(frame, f'Authentication', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
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
    if image is not None and image.size > 0:
        pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imageHash = imagehash.phash(pilImage)
        return imageHash
    else:
        return None

def isInsideRectangle(smallX, smallY, smallW, smallH, largeX, largeY, largeW, largeH):
    if smallX >= largeX and smallY >= largeY:
        if smallX + smallW <= largeX + largeW and smallY + smallH <= largeY + largeH:
            return True
    return False

def detectionAndTracking(frame):

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(rgbFrame)
    data = {}
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
        

        if len(arrNotFaceTracking):
            for num in arrNotFaceTracking:
                if num not in arrFaceTracking:
                    del trackers[num]
        elif len(arrFaceTracking):
            for num in list(trackers.keys()):
                if num not in arrFaceTracking:
                    del trackers[num]
        else:
            print(f"No faces found in the frame!")

        for key, (bbox) in list(trackers.items()):
            x, y, w, h = bbox    

            new_w = int(w * 0.7)
            new_h = int(h * 0.7)
            new_x = int(x + (w - new_w) / 2)
            new_y = int(y + (h - new_h) / 2)
            data[key] = (new_x, new_y, new_w, new_h)

    return data

def genUniqueId(length=20):
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(random.choices(characters, k=length))
    return unique_id

def videoCapture():
    global checkTimeRecognition, trackingIdAssign, imageHash, setTrackerName, trackers, prev_frame_time, new_frame_time

    rtsp = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0"
    video = "output_video.avi"
    video2 = "output_video2.avi"
    video3 = "6863054282731021257.mp4"
    cap = cv2.VideoCapture(video3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)
    second = 0
    countSecond = 0
    if not cap.isOpened():
        textToSpeech("Không thể mở máy ảnh")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        originalFrame = frame.copy()
        # cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        detections = detectionAndTracking(frame)
        arrDetection = {}

        if len(detections):
            for key, (bbox) in list(detections.items()):

                trackerId = key
                x, y, w, h = bbox

                w = x + w
                h = y + h

                topPoint = int((x + w) / 2)
                # cv2.circle(frame, (topPoint, h), 3, color.Red, -1)
                checkPointLine = cv2.pointPolygonTest(points, (topPoint, h), False)
                checkPointLine = 1
                if checkPointLine > 0:
                    arrDetection[trackerId] = bbox
            if len(arrDetection) and checkTime(checkTimeRecognition, 0.2):
                arrTrackerId = sorted(arrDetection.keys()) if arrDetection is not None else []
                if trackingIdAssign is None:
                    trackingIdAssign = min(arrTrackerId, default=None)
                    
                if trackingIdAssign not in arrTrackerId:
                    trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)

                if trackingIdAssign not in setTrackerName:
                    setTrackerName[trackingIdAssign] = {"name": "Unknown", "employeeCode": None, "Unknown": 0, "Known": 0, "authenticate": "", "timekeeping": False, "timeChecked": 0, "timer": datetime.now(), "time": datetime.now()}
                
                if setTrackerName[trackingIdAssign]["timeChecked"] >= 0.5:
                    if setTrackerName[trackingIdAssign]["timekeeping"]:
                        trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                    else:
                        if setTrackerName[trackingIdAssign]["name"] != "Unknown":
                            percentage = round((setTrackerName[trackingIdAssign]['Unknown'] / setTrackerName[trackingIdAssign]['Known']) * 100, 2)
                            if percentage <= 25:
                                setTrackerName[trackingIdAssign]["timekeeping"] = True
                                x, y, w, h = arrDetection[trackingIdAssign]
                                updateInfo({setTrackerName[trackingIdAssign]['employeeCode']: originalFrame[y:y+h, x:x+w]})
                                trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                            else:
                                trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                        else:
                            trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                else:
                    isOther = False
                    x, y, w, h = arrDetection[trackingIdAssign]
                    dataHash = getImageHash(originalFrame[y:y+h, x:x+w])
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
                        getNameFace(originalFrame[y:y+h, x:x+w], trackingIdAssign)
                    setTrackerName[trackingIdAssign]["timeChecked"] += 0.2

                checkTimeRecognition = datetime.now()
            for key, (bbox) in list(detections.items()):
                x, y, w, h = bbox
                trackerName = "Unknown"
                isPass = False
                if key in setTrackerName:
                    isPass = setTrackerName[key]["timekeeping"]
                    if isPass:
                        trackerName = "Successful"
                    else:
                        trackerName = setTrackerName[key]["name"]
                        
                cv2.putText(frame, f'# {key} {trackerName}', (x, int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            trackers = {}
            setTrackerName = {}
            trackingIdAssign = None
            imageHash = None

        if second != datetime.now().second:
            second = datetime.now().second
            if countSecond != 0:
                fps = countSecond
            countSecond = 1
        else:
            countSecond+=1
        text = f'FPS: {fps} trackingId: {trackingIdAssign}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.8, 2
        textColor, bgColor = color.White, (167, 80, 167)

        (textWidth, textHeight), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 30, 30

        cv2.rectangle(frame, (x - 10, y - textHeight - 10), (x + textWidth + 10, y + 10), bgColor, -1)
        cv2.putText(frame, text, (x, y), font, scale, textColor, thickness)
        for name in [name for name in locals() if name not in ['self', 'request', 'response', 'app', '__name__']]:
            if locals()[name] is not None:
                memory_size = sys.getsizeof(locals()[name])
                print(f"Variable '{name}' is using {memory_size} bytes of memory.")
                # del locals()[name]
        print("------------------------------------")
        cv2.imshow("ByteTrack", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
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

# try:
# except Exception as e:
#     print(f"An error occurred: {e}")
#     textToSpeech("Lỗi phát hiện khuôn mặt, Vui lòng liên hệ Admin", 1.2)
# finally:
#     return data
