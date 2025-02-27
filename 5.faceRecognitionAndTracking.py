import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import time

with mp_face_detection.FaceDetection(model_selection=1) as face_detection:
    rtsp = "rtsp://admin:bvTTDCaps999@192.168.40.38:554/cam/realmonitor?channel=1&subtype=0"
    cap = cv2.VideoCapture(0)

    while True:
        start =  time.time()
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                print(f"detection: {detection}")
                mp_drawing.draw_detection(frame, detection)

        end = time.time() 
        totalTime = end - start
        fps = int(1 / totalTime)

        text = f'FPS: {fps}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.8, 2
        textColor, bgColor = (255, 255, 255), (167, 80, 167)

        (textWidth, textHeight), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 30, 30

        cv2.rectangle(frame, (x - 10, y - textHeight - 10), (x + textWidth + 10, y + 10), bgColor, -1)

        cv2.putText(frame, text, (x, y), font, scale, textColor, thickness)
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
