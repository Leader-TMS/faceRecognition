import cv2
import datetime
import os
import uuid
from ultralytics import YOLO
model = YOLO("yolo/yolov8s.pt")
# Tải các cascade xml đã huấn luyện
face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_mouth.xml') 
left_eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_lefteye.xml')
right_eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_righteye.xml')
eye_glasses_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye_tree_eyeglasses.xml')

def detectedFace(image_array):

    for img in image_array:

        gray_person_image = img
        # gray_person_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_person_image, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            x = int(x - 0.4 * w)
            y = int(y - 0.4 * h)
            w = int(w * 1.4)
            h = int(h * 1.4)
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)

            cropped_image = gray_person_image[y:y + h, x:x + w]
            
            roi_gray = gray_person_image[y:y + h, x:x + w]
            
            left_eye = left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            left_eye_detected = len(left_eye) > 0

            right_eye = right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
            right_eye_detected = len(right_eye) > 0

            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_detected = len(eyes) > 0

            eye_glasses = eye_glasses_cascade.detectMultiScale(roi_gray)
            eye_glasses_detected = len(eye_glasses) > 0

            noses = nose_cascade.detectMultiScale(roi_gray)
            nose_detected = len(noses) > 0 
            
            mouths = mouth_cascade.detectMultiScale(roi_gray)
            mouth_detected = len(mouths) > 0 
            if eye_detected or nose_detected or mouth_detected or left_eye_detected or right_eye_detected or eye_glasses_detected:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                target_file_name = f'stored-faces/{current_time}_face{i}.jpg'
                cv2.imwrite(target_file_name, cropped_image)
                print(f"Saved face {i}")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    person_array = []
    for box in results[0].boxes.data:
        if int(box[5]) == 0:
            x, y, w, h = map(int, box[:4])
            cropped_person = frame[y:y + h, x:x + w]
            if cropped_person.size > 0:
                random_name = str(uuid.uuid4()) + ".jpg"
                cv2.imwrite(os.path.join('detected_persons', random_name), cropped_person)
                person_array.append(cropped_person)
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    detectedFace(person_array)
    cv2.imshow("YOLOv8 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
