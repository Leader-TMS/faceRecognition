import cv2

rtsp_url = "rtsp://admin:bvTTDCaps999@118.69.244.149:554/Streaming/Channels/102/"
cap = cv2.VideoCapture(rtsp_url)
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
fps = 60
cap.set(cv2.CAP_PROP_FPS, fps)
if not cap.isOpened():
    print("Không thể kết nối tới RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc khung hình.")
        break

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

