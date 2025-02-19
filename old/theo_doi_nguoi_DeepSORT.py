import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
rtsp_url = "rtsp://admin:bvTTDCaps999@118.69.244.149:554/Streaming/Channels/102/"
model = YOLO("yolo/yolo11s.pt")
tracker = DeepSort(max_age=50)

cap = cv2.VideoCapture(rtsp_url)
CONFIDENCE_THRESHOLD = 0.8
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = model(frame)[0]
    results = []
    for box in detections.boxes.data.tolist():
        confidence = box[4]
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        class_id = int(box[5])
        results.append([[x, y, w - x, h - y], confidence, class_id])
        tracks = tracker.update_tracks(results, frame=frame)
        for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

                x, y, w, h = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                cv2.rectangle(frame, (x, y), (w, h), GREEN, 2)
                cv2.putText(frame, f"Person: {str(track_id)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # cv2.rectangle(frame, (x, y) , (w, h), GREEN, 2)
        # cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
