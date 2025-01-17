from multiprocessing import Queue, Process
import cv2
import mediapipe as mp
import time
import supervision as sv
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import numpy as np
from datetime import datetime
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

# Setup Yolo and Yolo Face
model = YOLO("yolo/yolo11s.pt")
model.verbose = False
tracker = sv.ByteTrack()
boundingBoxAnnotator = sv.BoxAnnotator()
labelAnnotator = sv.LabelAnnotator()
# -------------------------------------
modelFace = YOLO("yolo/yolov11s-face.pt")
modelFace.verbose = False
mtcnn = MTCNN(keep_all=True)
inception_model = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs 
               if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=1"
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

        # play(audio)
        
    thread = threading.Thread(target=generateAndPlayAudio)
    thread.daemon=True
    thread.start()

def updateInfo(rfid = "", file_name = None, user_name = None):
    try:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        speech = threading.Thread(target=textToSpeech("Không tìm thấy tập tin!", 1.2))
        speech.daemon = True
        speech.start()
        return False
        
    if user_name is not None:
        for key, value in data.items():
            if value["name"] == user_name:
                rfid = key
                break

        if rfid == "":
            speech = threading.Thread(target=textToSpeech("Người dùng chưa được đăng ký trong hệ thống.", 1.2))
            speech.daemon = True
            speech.start()
            return False
        
    if rfid is not None:
        if rfid not in data:
            speech = threading.Thread(target=textToSpeech("Người dùng chưa được đăng ký trong hệ thống.", 1.2))
            speech.daemon = True
            speech.start()
            return False
        
    # if  data[rfid]['name'] != user_name:
    #     speech = threading.Thread(target=textToSpeech(f"khác.", 1.3))
    #     # speech = threading.Thread(target=textToSpeech(f"Tên đăng ký thẻ ({data[rfid]['name']}) khác với tên đăng ký khuôn mặt ({user_name}).", 1.3))
    #     speech.daemon = True
    #     speech.start()
    #     return False
    
    data[rfid]['total_scan'] += 1
    
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    data[rfid]['scanning_time'].append(current_time)

    with open(file_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    speech = threading.Thread(target=textToSpeech(f"Xin chào {data[rfid]['name']}", 1.2))
    speech.daemon = True
    speech.start()
    return data[rfid]['name']

def saveFaceDirection(image, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
    target_file_name = os.path.join(folderName, f'{current_time}.jpg')
    cv2.imwrite(target_file_name, image)

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

def faceRecognition(frame):
    results = modelFace(frame)
    label = "Unknown"
    for i, box in enumerate(results[0].boxes.data):
        x, y, w, h = map(int, box[:4])

        faces = frame[y - 10:h + 10, x - 10:w + 10]
        try:
            faces_mtcnn = mtcnn(faces)
            if faces_mtcnn is not None:
                for face in faces_mtcnn:
                    embedding = inception_model(face.unsqueeze(0)).detach().numpy().flatten()
                    embedding = np.array([embedding])
                    label_index = svm_model.predict(embedding)[0]
                    prob = svm_model.predict_proba(embedding)[0]
                    prob_percent = prob[label_index] * 100
                    if prob_percent >= 90:
                        label = label_encoder.inverse_transform([label_index])[0]
        except RuntimeError as e:
            print(f"Warning: {e}. Skipping this face region.")
    return label

def getNameFace(frame, trackerId):
    if frame.size > 0:
        global inputRFID, seeRFID, userAreCheckIn
        if trackerId not in setTrackerName:
            setTrackerName[trackerId] = {"name": "Unknown", "reCheckFace": 0, "authenticate": "", "numAuthenticate": 0}
        if setTrackerName[trackerId]["numAuthenticate"] > -40:
            if seeRFID:
                trackerName = saveEvidence(frame, trackerId)
                if isinstance(trackerName, str):
                    setTrackerName[trackerId]["name"] = trackerName
                    setTrackerName[trackerId]["numAuthenticate"] +=1
            elif setTrackerName[trackerId]["name"] == "Unknown":
                    trackerName = faceRecognition(frame)
                    if isinstance(trackerName, str) and trackerName != "Unknown":
                        setTrackerName[trackerId]["name"] = trackerName
                        if setTrackerName[trackerId]["numAuthenticate"] < 0:
                            setTrackerName[trackerId]["numAuthenticate"] = 0
                        setTrackerName[trackerId]["numAuthenticate"] +=1
                        saveEvidence(frame, trackerId, trackerName)
                    else:
                        setTrackerName[trackerId]["numAuthenticate"] -=1
            elif setTrackerName[trackerId]["reCheckFace"] >= fps * 2 and setTrackerName[trackerId]["numAuthenticate"] < 3:
                trackerName = faceRecognition(frame)
                if isinstance(trackerName, str) and trackerName != "Unknown":
                    if setTrackerName[trackerId]["name"] == trackerName:
                        setTrackerName[trackerId]["numAuthenticate"] +=1
                    else:
                        setTrackerName[trackerId]["numAuthenticate"] -=1
                        saveEvidence(frame, trackerId, trackerName)
                    setTrackerName[trackerId]["name"] = trackerName
                else:
                    setTrackerName[trackerId]["name"] = "Unknown"
                setTrackerName[trackerId]["reCheckFace"] = 0
            elif setTrackerName[trackerId]["numAuthenticate"] < 3:
                setTrackerName[trackerId]["reCheckFace"]+=1
        elif seeRFID:
            trackerName = saveEvidence(frame, trackerId)
            if isinstance(trackerName, str):
                setTrackerName[trackerId]["name"] = trackerName
        print(setTrackerName[trackerId])

def scanRFID():
    global inputRFID, seeRFID
    while True:
        k = readkey()
        with lock:
            if k not in [key.ENTER, key.BACKSPACE]:
                inputRFID += k
            if k == key.ENTER:
                seeRFID = True
            if k == key.BACKSPACE:
                inputRFID = inputRFID[:-1]
def videoCapture():
    while cap.isOpened():
        start =  time.time()
        ret, frame = cap.read()
        if not ret:
            break

        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)

        if len(detections) > 0:
            for i, detection in enumerate(detections):
                trackerId = detections.tracker_id[i]
                x, y, w, h = map(int, detection[0])
                topPoint = int((x + w) / 2)
                cv2.circle(frame, (topPoint, y - 40), 3, color.Red, -1)
                if detections.confidence[i] >= 0.5:
                    checkPointLine = cv2.pointPolygonTest(points, (topPoint, y), False)
                    if checkPointLine > 0:
                        cv2.putText(frame, f'getNameFace', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        getNameFace(frame[y:h, x:w], trackerId)
            labels = [
                f"# {tracker_id} {setTrackerName[tracker_id]['name']} {confidence:0.2f}" if tracker_id in setTrackerName else f"# {tracker_id} Unknown {confidence:0.2f}"
                for confidence, tracker_id
                in zip(detections.confidence, detections.tracker_id)
            ]
            annotatedFrame = boundingBoxAnnotator.annotate(scene=frame.copy(), detections=detections)
            annotatedFrame = labelAnnotator.annotate(scene=annotatedFrame, detections=detections, labels=labels)

        else:
            annotatedFrame = frame
        end = time.time() 
        totalTime = end - start

        fps = 1 / totalTime
        cv2.putText(annotatedFrame, f'FPS: {int(fps)} inputRFID:{inputRFID}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
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