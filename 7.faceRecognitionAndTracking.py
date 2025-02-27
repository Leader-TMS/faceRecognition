import cv2

# Kiểm tra phiên bản OpenCV
print(cv2.__version__)

# Tạo danh sách chứa các tracker
trackers = cv2.legacy.MultiTracker_create()
rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"

# Mở camera hoặc video
cap = cv2.VideoCapture(rtsp)

# Đọc khung hình đầu tiên
ret, frame = cap.read()

if not ret:
    print("Không thể mở camera hoặc video.")
    exit()

# Hiển thị khung hình và cho phép người dùng chọn đối tượng
while True:
    # Chọn các đối tượng (nhiều đối tượng)
    bboxes = cv2.selectROIs("Select Objects", frame, fromCenter=False, showCrosshair=True)

    if len(bboxes) > 0:
        for bbox in bboxes:
            # Thêm từng tracker vào MultiTracker
            tracker = cv2.legacy.TrackerKCF_create()  # Sử dụng cv2.legacy.TrackerKCF_create()
            trackers.add(tracker, frame, bbox)

        break  # Chọn xong, thoát vòng lặp để bắt đầu theo dõi

# Vòng lặp xử lý từng khung hình
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Cập nhật tất cả các tracker
    success = trackers.update(frame)  # success là một giá trị boolean, không phải danh sách

    # Nếu thành công, vẽ bounding box
    if success:
        for i, bbox in enumerate(trackers.getObjects()):
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Tracking", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
