from ultralytics import YOLO
import cv2
import datetime
model = YOLO("yolo/yolov11m-face.pt")

# cap = cv2.VideoCapture(0)
# width = 1920
# height = 1080
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Điều chỉnh exposure_auto (0: tắt tự động, 1: bật tự động)
# cap.set(cv2.CAP_PROP_EXPOSURE, -10)         # Điều chỉnh exposure_absolute (mức độ phơi sáng)
# cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)  # Điều chỉnh white_balance_temperature (Bấm hoặc giảm giá trị)
# cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)   # Điều chỉnh giá trị màu đỏ trong cân bằng trắng (tùy theo webcam)
# cap.set(cv2.CAP_PROP_FOCUS, 50)            # Điều chỉnh focus_absolute (Điều chỉnh tiêu cự)
# cap.set(cv2.CAP_PROP_SHARPNESS, 150)       # Điều chỉnh sharpness (Độ sắc nét)
# cap.set(cv2.CAP_PROP_ZOOM, 100) 
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# fps = 60
# cap.set(cv2.CAP_PROP_FPS, fps)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

def captureFace(frame, window_name):
    results = model(frame)

    for i, box in enumerate(results[0].boxes.data):
        x, y, w, h = map(int, box[:4])
        face = frame[y - 10:h + 10, x - 10:w + 10]
        if face is not None and face.size > 0:
            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
            target_file_name = f'stored-faces/{current_time}_face{i}.jpg'
            cv2.imwrite(target_file_name, face)
            print(f"Saved face {i}")
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        captureFace(frame1, "Capture face 1")
        captureFace(frame2, "Capture face 2")

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
