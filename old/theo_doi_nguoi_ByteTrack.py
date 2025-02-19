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
import compareFeaturePoints
import threading
from readchar import readkey, key
import json

mpFaceMesh = mp.solutions.face_mesh
stopThreading = False
inputRFID = ""
seeRFID = False
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
boundingBoxAnnotator = sv.BoundingBoxAnnotator()
labelAnnotator = sv.LabelAnnotator()

rtspUrl = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)
cap.set(cv2.CAP_PROP_FOCUS, 50)
cap.set(cv2.CAP_PROP_SHARPNESS, 150)
cap.set(cv2.CAP_PROP_ZOOM, 100)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
    target_file_name = f'{folderName}{current_time}_face.jpg'
    cv2.imwrite(target_file_name, image)
    print(f"Saved face")
    
def getNameFace(frame, trackerId, faceMesh):
    global inputRFID, seeRFID
    if trackerId not in setTrackerName:
        setTrackerName[trackerId] = {"name": "Unknown", "reCheckFace": 0, "direction": [], "authenticate": ""}
    if setTrackerName[trackerId]["name"] == "Unknown":
        # compare similarities

        # compareFeaturePoints.getBestMatchFolder(frame, 'face_direction')
        # bestFolder, match_count, match_percentage = compareFeaturePoints.getBestMatchFolder(frame, 'face_direction')
        # print(f"Thư mục con có sự tương đồng cao nhất: {bestFolder} với {match_count} điểm trùng khớp.")
        # print(f"Tỷ lệ trùng khớp: {match_percentage:.2f}%")
        # if isinstance(bestFolder, str) and bestFolder != "Unknown":
        #     setTrackerName[trackerId]["name"] = bestFolder
        # else:
        trackerName, _ = faceRecognition.faceRecognition(frame, trackerId)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
    elif setTrackerName[trackerId]["reCheckFace"] >= fps * 3:
        trackerName, _ = faceRecognition.faceRecognition(frame, trackerId)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
        setTrackerName[trackerId]["reCheckFace"] = 0
    # compare similarities
    
    # elif len(setTrackerName[trackerId]["direction"]) < 3:
    #     trackerName, headTiltPose = faceRecognition.faceRecognition(frame, trackerId, faceMesh)
    #     if isinstance(trackerName, str) and trackerName != "Unknown":
    #         direction = setTrackerName[trackerId]["direction"]
    #         if headTiltPose in ["Left", "Right", "Down"]:
    #             if headTiltPose not in direction:
    #                 now = datetime.datetime.now().strftime("%Y-%m-%d")
    #                 saveFaceDirection(frame, f'face_direction/{trackerName}_{now}/')
    #                 setTrackerName[trackerId]["direction"].append(headTiltPose)
    #     setTrackerName[trackerId]["reCheckFace"]+=1
    else:
        setTrackerName[trackerId]["reCheckFace"]+=1
    print(setTrackerName[trackerId])
    if seeRFID and setTrackerName[trackerId]["name"] != "Unknown":
        nameInRFID = updateInfo(inputRFID, "rfid_data.json", setTrackerName[trackerId]["name"])
        if not nameInRFID:
            setTrackerName[trackerId]["authenticate"] = "Invalid RFID info"
        else:
            setTrackerName[trackerId]["authenticate"] = "Valid RFID info"
        seeRFID = False
        inputRFID = ""

def updateInfo(rfid, file_name, user_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("File not found!")
        return False
    
    if rfid not in data:
        print(f"RFID {rfid} không tồn tại trong dữ liệu.")
        return False

    if data[rfid]['name'] != user_name:
        return False

    data[rfid]['total_scan'] += 1
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if data[rfid]['check_scan'] == 0:
        data[rfid]['start_time'] = current_time
        data[rfid]['check_scan'] = 1
        print(f"Đã cập nhật thời gian bắt đầu cho {rfid}: {current_time}")
    elif data[rfid]['check_scan'] == 1:
        data[rfid]['end_time'] = current_time
        data[rfid]['check_scan'] = 0
        print(f"Đã cập nhật thời gian kết thúc cho {rfid}: {current_time}")

    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print("Dữ liệu đã được lưu lại!")
    return True

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
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = tracker.update_with_detections(detections)
            labels = []
            for i, detection in enumerate(detections):
                trackerId = detections.tracker_id[i]
                x1, y1, x2, y2 = map(int, detection[0])
                getNameFace(frame[y1:y2, x1:x2], trackerId, face_mesh)
                labels.append(setTrackerName[trackerId]["name"])
                
            #     endpointPersonX = int((x1 + x2) / 2)
            #     cv2.circle(annotated_frame, (endpointPersonX, y2), 5, color.Yellow, -1) 
            #     if trackerId not in locationTracking:
            #         locationTracking[trackerId] = {"y": 0, "direction": "top", "checkDirection": 0 }
            #     print(f'trackerId: {trackerId} - {endpointPersonX >= roi_x1} - {y2 >= roi_y1} - {locationTracking[trackerId]["y"] < y2}')
            #     if(endpointPersonX >= roi_x1 and y2 >= roi_y1 and locationTracking[trackerId]["y"] < y2):
            #         locationTracking[trackerId]["checkDirection"] = locationTracking[trackerId]["checkDirection"] + 1
            #         locationTracking[trackerId]["y"] = y2
            #         print(f'trackerId: {trackerId} - {locationTracking}')
            #     elif(locationTracking[trackerId]["checkDirection"] > 0 and endpointPersonX >= roi_x1 and y2 >= roi_y1 and locationTracking[trackerId]["y"] > y2):
            #         locationTracking[trackerId]["checkDirection"] = locationTracking[trackerId]["checkDirection"] - 1
            #         locationTracking[trackerId]["y"] = y2
            #     if (locationTracking[trackerId]["checkDirection"] >= 10):
            #         locationTracking[trackerId]["direction"] = "bottom"
            #         cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), color.White, 2)

            annotated_frame = boundingBoxAnnotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = labelAnnotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imshow("ByteTrack", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopThreading = True
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    keyboard_thread = threading.Thread(target=scanRFID)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    video_capture_thread = threading.Thread(target=videoCapture)
    video_capture_thread.daemon = True
    video_capture_thread.start()

    keyboard_thread.join()
    video_capture_thread.join()