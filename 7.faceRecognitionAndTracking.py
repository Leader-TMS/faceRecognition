import cv2

# Kiểm tra phiên bản OpenCV
print(cv2.__version__)

# Tạo danh sách chứa các tracker
trackers = cv2.MultiTracker_create()
rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if not ret:
    print("Không thể mở camera hoặc video.")
    exit()

while True:
    bboxes = cv2.selectROIs("Select Objects", frame, fromCenter=False, showCrosshair=True)

    if len(bboxes) > 0:
        for bbox in bboxes:
            tracker = cv2.TrackerKCF_create()
            trackers.add(tracker, frame, bbox)
        break

while True:
    ret, frame = cap.read()
    if not ret:
        break
    success = trackers.update(frame)

    if success:
        for i, bbox in enumerate(trackers.getObjects()):
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
