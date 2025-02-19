import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter
from osnet_pytorch import OSNet
import supervision as sv
import configparser
from ultralytics import YOLO

config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]

model = YOLO("yolo/yolo11l.pt")
tracker = sv.ByteTrack()

kf_dict = {}

osnet = OSNet(pretrained=True)

camera_paths = ["camera_1.mp4", "camera_2.mp4"]
caps = [cv2.VideoCapture(path) for path in camera_paths]

def extract_reid_features(crop_img):
    crop_img = cv2.resize(crop_img, (128, 256))
    crop_img = crop_img / 255.0
    crop_img = np.transpose(crop_img, (2, 0, 1))
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = torch.from_numpy(crop_img).float()
    with torch.no_grad():
        features = osnet(crop_img)
    return features.squeeze().cpu().numpy()

def track_objects_across_cameras(caps):
    active_tracks = {}

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) != len(caps):
            break
        all_detections = []
        for frame in frames:
            results = model(frame)
            detections = results.xywh[0].cpu().numpy()
            boxes = []
            for det in detections:
                if det[4] > 0.5:
                    x_center, y_center, w, h = det[:4]
                    x1 = int((x_center - w / 2))
                    y1 = int((y_center - h / 2))
                    x2 = int((x_center + w / 2))
                    y2 = int((y_center + h / 2))
                    boxes.append([x1, y1, x2, y2])
            all_detections.append(boxes)
        tracker.update(all_detections)
        for track in tracker.active_tracks():
            track_id = track.track_id
            track_box = track.to_tlbr()

            if track_id not in kf_dict:
                kf_dict[track_id] = KalmanFilter(dim_x=4, dim_z=2)
                kf_dict[track_id].F = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]])
                kf_dict[track_id].H = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]])

            kf_dict[track_id].predict()
            kf_dict[track_id].update([track_box[0], track_box[1]])

            object_crop = frames[0][track_box[1]:track_box[3], track_box[0]:track_box[2]]
            object_features = extract_reid_features(object_crop)

            for prev_track_id, prev_features in active_tracks.items():
                if prev_track_id != track_id:
                    similarity = np.dot(object_features, prev_features.T)
                    if similarity > 0.8:
                        print(f"Đối tượng {track_id} trùng với {prev_track_id}")

            active_tracks[track_id] = object_features

            cv2.rectangle(frames[0], (int(track_box[0]), int(track_box[1])),
                          (int(track_box[2]), int(track_box[3])), (0, 255, 0), 2)
            cv2.putText(frames[0], f"ID: {track_id}", (int(track_box[0]), int(track_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Tracking - Camera 1", frames[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

track_objects_across_cameras(caps)
