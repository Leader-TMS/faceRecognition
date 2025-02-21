import cv2
rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(rtsp)

if not cap.isOpened():
    print("Không thể kết nối tới RTSP stream")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video2.avi', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Không nhận được frame, thoát...")
        break

    out.write(frame)

    cv2.imshow("RTSP Stream", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()