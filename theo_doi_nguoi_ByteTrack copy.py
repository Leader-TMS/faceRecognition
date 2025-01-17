import cv2
import torch
from ultralytics import YOLO
import supervision as sv
import configparser
import faceRecognition
from filterpy.kalman import KalmanFilter
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]

model = YOLO("yolo/yolo11l.pt")
tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"
cap = cv2.VideoCapture(0)
width = 720
height = 480
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -10)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)
cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)
cap.set(cv2.CAP_PROP_FOCUS, 50)
cap.set(cv2.CAP_PROP_SHARPNESS, 150)
cap.set(cv2.CAP_PROP_ZOOM, 100)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 60)

CONFIDENCE_THRESHOLD = 0.8
roi_x1, roi_y1, roi_x2, roi_y2 = 250, 250, 600, 350
locationTracking = {}
setTrackerName = {} 

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

color = Color()

kalman_filters = {}

def create_kalman_filter():
    kalman = KalmanFilter(dim_x=4, dim_z=2)
    kalman.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])  # Ma trận chuyển động
    kalman.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])  # Ma trận quan sát
    kalman.P *= 1000.  # Độ không chắc chắn ban đầu
    kalman.R = np.array([[10, 0], [0, 10]])  # Độ nhiễu đo lường
    kalman.Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Độ nhiễu của quá trình
    return kalman

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

        if trackerId not in kalman_filters:
            kalman_filters[trackerId] = create_kalman_filter()

        kalman = kalman_filters[trackerId]
        kalman.predict()
        kalman.update([x1, y1])

        predicted_position = kalman.x[:4]
        x_pred, y_pred, w_pred, h_pred = int(predicted_position[0]), int(predicted_position[1]), int(predicted_position[2]), int(predicted_position[3])
        if trackerId not in setTrackerName or setTrackerName[trackerId]["name"] == "Unknown":
            trackerName = faceRecognition.faceRecognition(frame[y1:y2, x1:x2], trackerId)
            if isinstance(trackerName, str) and trackerName != "Unknown":
                setTrackerName[trackerId] = {"name": trackerName}
            else:
                setTrackerName[trackerId] = {"name": "Unknown"}
        
        labels.append(setTrackerName[trackerId]["name"])

        cv2.rectangle(frame, (x_pred, y_pred), (x2, y2), color.Green, 2)
        cv2.putText(frame, setTrackerName[trackerId]["name"], (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.Yellow, 2)
        # cv2.putText(frame, f"ID: {trackerId} Pred: ({x_pred}, {y_pred})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.Yellow, 2)

    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    cv2.imshow("ByteTrack with Kalman", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
