import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(3)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
if not cap1.isOpened():
    print("Không thể mở camera")
else:
    while True:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()

        if not ret:
            break
        cv2.imshow('Frame1', frame1)
        cv2.imshow('Frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cv2.destroyAllWindows()
# 1024x768 = 640x480