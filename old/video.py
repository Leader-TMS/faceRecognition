import cv2
import time
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

# cap.set(3, 720)
# cap.set(4, 480)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (frame_width, frame_height))

while True:
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể đọc video.")
        break

    out.write(frame)
    cv2.imshow('Video', frame)
    elapsed_time = time.time() - start_time
    wait_time = max(1.0 / 60 - elapsed_time, 0)
    print(wait_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
