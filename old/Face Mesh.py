import cv2
import mediapipe as mp
import numpy as np

mpFaceMesh = mp.solutions.face_mesh
mpDrawing = mp.solutions.drawing_utils

faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

import numpy as np

def goodFaceAngle(nose, leftEye, rightEye, fWidth, fHeight):

    noseX, noseY = int(nose.x * fWidth), int(nose.y * fHeight)
    leftEyeX, leftEyeY = int(leftEye.x * fWidth), int(leftEye.y * fHeight)
    rightEyeX, rightEyeY = int(rightEye.x * fWidth), int(rightEye.y * fHeight)
    
    vector1 = np.array([leftEyeX - noseX, leftEyeY - noseY])

    vector2 = np.array([rightEyeX - noseX, rightEyeY - noseY])

    dotProduct = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosAngle = round(dotProduct / (magnitude1 * magnitude2), 2)

    return cosAngle <= 0.7


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgbFrame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            leftEye = landmarks.landmark[173]
            rightEye = landmarks.landmark[398]
            nose = landmarks.landmark[4]
            angle = goodFaceAngle(nose, leftEye, rightEye, 640, 480)

            if angle:
                cv2.putText(frame, 'Good Face', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Face is not good', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame, f'Angle: {angle}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Rotation Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np

# mpFaceMesh = mp.solutions.face_mesh
# mpDrawing = mp.solutions.drawing_utils

# faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = faceMesh.process(rgbFrame)

#     if results.multi_face_landmarks:
#         for landmarks in results.multi_face_landmarks:
#             for i, landmark in enumerate(landmarks.landmark):
#                 x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

#                 cv2.putText(frame, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

#     cv2.imshow('Face Mesh with Landmark Numbers', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
