import cv2
import torch
import os
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import supervision as sv
import configparser
import faceRecognition
import mediapipe as mp
import datetime
import threading
from readchar import readkey, key
import json
from textToSpeech import textToSpeechVietnamese
import time
import multiprocessing

mpFaceMesh = mp.solutions.face_mesh
stopThreading = False
inputRFID = ""
seeRFID = False
userAreCheckIn = ""

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

config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]

model = YOLO("yolo/yolo11l.pt")
model.verbose = False
tracker = sv.ByteTrack()
boundingBoxAnnotator = sv.BoxAnnotator()
labelAnnotator = sv.LabelAnnotator()

rtspUrl = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"
cap = cv2.VideoCapture(3)
width = 1024
height = 768
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)
cap.set(cv2.CAP_PROP_FOCUS, 50)
cap.set(cv2.CAP_PROP_SHARPNESS, 150)
cap.set(cv2.CAP_PROP_ZOOM, 100)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)
fps = cap.get(cv2.CAP_PROP_FPS)

roi_x1, roi_y1, roi_x2, roi_y2 = 250, 250, 600, 350

locationTracking = {}
setTrackerName = {} 
color = Color()
def saveFaceDirection(image, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
        print(f"Created directory: {folderName}")
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
    target_file_name = os.path.join(folderName, f'{current_time}.jpg')
    cv2.imwrite(target_file_name, image)
    print(f"Saved face")

def saveEvidence(frame, trackerId, trackerName = None):
    global inputRFID, seeRFID
    nameInRFID = updateInfo(inputRFID, "rfid_data.json", trackerName)
    if not nameInRFID:
        setTrackerName[trackerId]["authenticate"] = "Invalid RFID info"
        saveFaceDirection(frame, "evidences/invalid/" + setTrackerName[trackerId]["name"])
    else:
        setTrackerName[trackerId]["name"] = nameInRFID
        setTrackerName[trackerId]["authenticate"] = "Valid RFID info"
        saveFaceDirection(frame, "evidences/valid/" + setTrackerName[trackerId]["name"])
    seeRFID = False
    inputRFID = ""
    return nameInRFID

def getNameFace(frame, trackerId, faceMesh):
    global inputRFID, seeRFID, userAreCheckIn
    if trackerId not in setTrackerName:
        setTrackerName[trackerId] = {"name": "Unknown", "reCheckFace": 0, "authenticate": ""}
    if setTrackerName[trackerId]["name"] == "Unknown":
        if seeRFID:
            trackerName = saveEvidence(frame, trackerId)
            if isinstance(trackerName, str):
                setTrackerName[trackerId]["name"] = trackerName
        else:
            trackerName, _ = faceRecognition.faceRecognition(frame, trackerId)
            if isinstance(trackerName, str) and trackerName != "Unknown":
                setTrackerName[trackerId]["name"] = trackerName
                if trackerName != userAreCheckIn:
                    saveEvidence(frame, trackerId, trackerName)
                    userAreCheckIn = trackerName

    elif setTrackerName[trackerId]["reCheckFace"] >= fps * 2:
        trackerName, _ = faceRecognition.faceRecognition(frame, trackerId)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
            if trackerName != userAreCheckIn:
                userAreCheckIn = trackerName
                saveEvidence(frame, trackerId, trackerName)
        else:
            setTrackerName[trackerId]["name"] = "Unknown"
        setTrackerName[trackerId]["reCheckFace"] = 0
    else:
        if seeRFID:
            trackerName = saveEvidence(frame, trackerId, setTrackerName[trackerId]["name"])
            if isinstance(trackerName, str):
                setTrackerName[trackerId]["name"] = trackerName
        setTrackerName[trackerId]["reCheckFace"]+=1

    # if seeRFID and setTrackerName[trackerId]["name"] != "Unknown":
    #     nameInRFID = updateInfo(inputRFID, "rfid_data.json", setTrackerName[trackerId]["name"], )
    #     if not nameInRFID:
    #         setTrackerName[trackerId]["authenticate"] = "Invalid RFID info"
    #         saveFaceDirection(frame, "evidences/invalid/" + setTrackerName[trackerId]["name"])
    #     else:
    #         setTrackerName[trackerId]["authenticate"] = "Valid RFID info"
    #         saveFaceDirection(frame, "evidences/valid/" + setTrackerName[trackerId]["name"])
    # elif seeRFID and setTrackerName[trackerId]["name"] == "Unknown":    

    # elif seeRFID:
    #     seeRFID = False
    #     inputRFID = None
    print(setTrackerName[trackerId])
    

def updateInfo(rfid = "", file_name = None, user_name = None):
    try:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        speech = threading.Thread(target=textToSpeechVietnamese("Không tìm thấy tập tin!", 1.2))
        speech.daemon = True
        speech.start()
        return False
    found = False

        
    if user_name is not None and rfid == "":
        for key, value in data.items():
            if value["name"] == user_name:
                rfid = key
                break

        if rfid == "":
            speech = threading.Thread(target=textToSpeechVietnamese("Người dùng chưa được đăng ký trong hệ thống 1.", 1.2))
            speech.daemon = True
            speech.start()
            return False
        
    if rfid is not None:
        if rfid not in data:
            speech = threading.Thread(target=textToSpeechVietnamese("Người dùng chưa được đăng ký trong hệ thống 2.", 1.2))
            speech.daemon = True
            speech.start()
            return False
        
    if  data[rfid]['name'] != user_name:
        speech = threading.Thread(target=textToSpeechVietnamese(f"Tên đăng ký thẻ ({data[rfid]['name']}) khác với tên đăng ký khuôn mặt ({user_name}).", 1.3))
        speech.daemon = True
        speech.start()
        return False
    
    data[rfid]['total_scan'] += 1
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    data[rfid]['scanning_time'].append(current_time)

    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    speech = threading.Thread(target=textToSpeechVietnamese(f"Xin chào {data[rfid]['name']}", 1.2))
    speech.daemon = True
    speech.start()
    return data[rfid]['name']

def scanRFID():
    global inputRFID, seeRFID
    while not stopThreading:
        k = readkey()
        if k not in [key.ENTER, key.BACKSPACE]:
            inputRFID += k
        if k == key.ENTER:
            seeRFID = True
        if k == key.BACKSPACE:
            inputRFID = inputRFID[:-1]
            print(f"Current input: {inputRFID}")
            
def videoCapture():
    global stopThreading
    with mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            start =  time.time()
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = tracker.update_with_detections(detections)
            labels = []
            if len(detections) > 0:
                for i, detection in enumerate(detections):
                    trackerId = detections.tracker_id[i]
                    x1, y1, x2, y2 = map(int, detection[0])
                    getNameFace(frame[y1:y2, x1:x2], trackerId, face_mesh)
                    labels.append(setTrackerName[trackerId]["name"])

                annotatedFrame = boundingBoxAnnotator.annotate(scene=frame.copy(), detections=detections)
                annotatedFrame = labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)
            else:
                annotatedFrame = frame
            end = time.time() 
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(annotatedFrame, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow("ByteTrack", annotatedFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopThreading = True
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
    # processes = []
    # process = multiprocessing.Process(target=scanRFID)
    # process.daemon = True
    # processes.append(process)
    # process.start()
    # process = multiprocessing.Process(target=videoCapture)
    # process.daemon = True
    # processes.append(process)
    # process.start()
    # # processes = []
    # # process = multiprocessing.Process(target=videoCapture)
    # # process.daemon = True
    # # processes.append(process)
    # # process.start()
    # for process in processes:
    #     process.join()

    # print("All processes finished.")