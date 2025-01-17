from ultralytics import YOLO
import cv2
rtsp_url = "rtsp://admin:bvTTDCaps999@118.69.244.149:554/Streaming/Channels/102/"
model = YOLO("yolo/yolo11l.pt")
cap = cv2.VideoCapture(0)
width = 720
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
fps = 60
cap.set(cv2.CAP_PROP_FPS, fps)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes.data:
        if int(box[5]) == 0:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
