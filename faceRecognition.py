import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import joblib
import time
import os
import json
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
import datetime
import time
import threading
from readchar import readkey, key
from g_helper import bgr2rgb, rgb2bgr, mirrorImage
from fp_helper import pipelineHeadTiltPose

seeRFID = False
model = YOLO("yolo/yolov11s-face.pt")
model.verbose = False
stop_flag = False
inputRFID = ""

mtcnn = MTCNN(keep_all = False)
inception_model = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def saveFaceUnknown(faces):
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
    target_file_name = f'unknown-faces/{current_time}_face.jpg'
    cv2.imwrite(target_file_name, faces)
    print(f"Saved face")

def faceRecognition(frame, trackerId = None, faceMesh = None):
    results = model(frame)
    print(len(results[0].boxes.data))
    for i, box in enumerate(results[0].boxes.data):
        x, y, w, h = map(int, box[:4])
        faces = frame[y - 10:h + 10, x - 10:w + 10]
        try:
            faces = mtcnn(faces)
            if faces is not None:
            #     for face in faces_mtcnn:
                    print(f"face: {faces}")
                    embedding = inception_model(faces.unsqueeze(0)).detach().numpy().flatten()
                    embedding = np.array([embedding])
                    label_index = svm_model.predict(embedding)[-1]
                    label = label_encoder.inverse_transform([label_index])[-1]
                    
                    prob = svm_model.predict_proba(embedding)[0]
                    prob_percent = prob[label_index] * 100
                    if prob_percent < 90:
                        label = "Unknown"
                    #     if faces is not None and faces.size > 0:
                    #         saveFaceUnknown(faces)
                    # print(f'label: {label}')
                    if trackerId is not None:
                        head_tilt_pose = None
                        if faceMesh is not None:
                            face_direction = mirrorImage(faces)
                            results = faceMesh.process(bgr2rgb(face_direction))
                            if results.multi_face_landmarks:
                                head_tilt_pose = pipelineHeadTiltPose(face_direction, results.multi_face_landmarks[0])
                        return label, head_tilt_pose
                    prob_text = f"{prob_percent:.2f}%"
                
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {label_index} ({prob_text})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # else:
            #     if faces is not None and faces.size > 0:
            #         saveFaceUnknown(faces)

        except RuntimeError as e:
            print(f"Warning: {e}. Skipping this face region.")

    if trackerId is not None:
        return None, None
    return frame

def keyboard_input():
    global inputRFID, seeRFID
    while not stop_flag:
        k = readkey()
        if k not in [key.ENTER, key.BACKSPACE]:
            inputRFID += k
        if k == key.ENTER:
            seeRFID = True
        if k == key.BACKSPACE:
            inputRFID = inputRFID[:-1]
            print(f"Current input: {inputRFID}")

def updateInfo(rfid, file_name='rfid_data.json'):
    try:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        print("File not found!")
        return
    
    if rfid not in data:
        print(f"RFID {rfid} không tồn tại trong dữ liệu.")
        return

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
    return data[rfid]['name']

def video_capture():
    global stop_flag
    rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:81/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(rtsp)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Điều chỉnh exposure_auto (0: tắt tự động, 1: bật tự động)
    cap.set(cv2.CAP_PROP_EXPOSURE, -10)         # Điều chỉnh exposure_absolute (mức độ phơi sáng)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)  # Điều chỉnh white_balance_temperature (Bấm hoặc giảm giá trị)
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)   # Điều chỉnh giá trị màu đỏ trong cân bằng trắng (tùy theo webcam)
    cap.set(cv2.CAP_PROP_FOCUS, 50)            # Điều chỉnh focus_absolute (Điều chỉnh tiêu cự)
    cap.set(cv2.CAP_PROP_SHARPNESS, 150)       # Điều chỉnh sharpness (Độ sắc nét)
    cap.set(cv2.CAP_PROP_ZOOM, 100) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = faceRecognition(frame)
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            stop_flag = True
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    # keyboard_thread = threading.Thread(target=keyboard_input)
    # keyboard_thread.start()

    video_capture_thread = threading.Thread(target=video_capture)
    video_capture_thread.start()

    # keyboard_thread.join()
    video_capture_thread.join()
