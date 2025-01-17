import cv2
import numpy as np
import tensorflow as tf
import json
import os
import uuid

model = tf.keras.models.load_model("face_recognition_model.h5")
model_efficientnet = tf.keras.models.load_model("face_recognition_model_efficientnet.h5")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_mouth.xml') 
left_eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_lefteye.xml')
right_eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_mcs_righteye.xml')
eye_glasses_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        x = int(x - 0.4 * w)
        y = int(y - 0.4 * h)
        w = int(w * 1.4)
        h = int(h * 1.4)
        x = max(0, x)
        y = max(0, y)
        w = min(frame.shape[1] - x, w)
        h = min(frame.shape[0] - y, h)
        cropped_image = frame[y:y + h, x:x + w]
                    
        left_eye = left_eye_cascade.detectMultiScale(cropped_image, scaleFactor=1.1, minNeighbors=5)
        left_eye_detected = len(left_eye) > 0

        right_eye = right_eye_cascade.detectMultiScale(cropped_image, scaleFactor=1.1, minNeighbors=5)
        right_eye_detected = len(right_eye) > 0

        eyes = eye_cascade.detectMultiScale(cropped_image)
        eye_detected = len(eyes) > 0

        eye_glasses = eye_glasses_cascade.detectMultiScale(cropped_image)
        eye_glasses_detected = len(eye_glasses) > 0

        noses = nose_cascade.detectMultiScale(cropped_image)
        nose_detected = len(noses) > 0 
        
        mouths = mouth_cascade.detectMultiScale(cropped_image)
        mouth_detected = len(mouths) > 0 
        if eye_detected or nose_detected or mouth_detected or left_eye_detected or right_eye_detected or eye_glasses_detected:

            img_resized = cv2.resize(gray, (224, 224))
            img_resized = img_resized.reshape(1, 224, 224, 1).astype('float32') / 255.0
            prediction = model_efficientnet.predict(img_resized)
            label = np.argmax(prediction)
            print(f'[{counter}]label - {label}')
            predicted_name = label_map[str(label)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # random_name = str(uuid.uuid4()) + ".jpg"
            # cv2.imwrite(os.path.join('stored-faces', random_name), cropped_image)

    counter+=1
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
