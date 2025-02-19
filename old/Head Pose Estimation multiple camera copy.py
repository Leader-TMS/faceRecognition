import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Khởi tạo OpenCV để sử dụng webcam
cap = cv2.VideoCapture(0)

# Hàm tính góc quay của khuôn mặt
def calculate_face_angle(landmarks):
    # Lấy các điểm landmarks quan trọng để tính toán góc quay
    # Ví dụ: Lấy các điểm mắt trái và mắt phải (dự đoán các điểm 3D)
    left_eye = [landmarks[33].x, landmarks[33].y, landmarks[33].z]
    right_eye = [landmarks[133].x, landmarks[133].y, landmarks[133].z]
    
    # Tính toán góc giữa mắt trái và mắt phải trong không gian 3D
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    dz = right_eye[2] - left_eye[2]
    
    # Tính góc quay trên trục X, Y, Z
    angle_x = np.arctan2(dy, dz)
    angle_y = np.arctan2(dx, dz)
    angle_z = np.arctan2(dy, dx)
    
    return angle_x, angle_y, angle_z

# Hàm kiểm tra khuôn mặt có được chụp đầy đủ các góc độ không
def check_face_rotation(landmarks):
    # Tính toán các góc quay trên ba trục (X, Y, Z)
    angle_x, angle_y, angle_z = calculate_face_angle(landmarks)
    
    # Đặt ngưỡng cho góc quay, ví dụ: nếu góc quay trên trục Y (góc xoay trái/phải) vượt quá 45 độ, khuôn mặt có thể bị xoay đủ
    if abs(angle_y) > 0.5:  # 0.5 rad = khoảng 30 độ
        return True
    elif abs(angle_x) > 0.5:  # 0.5 rad = khoảng 30 độ
        return True
    elif abs(angle_z) > 0.5:  # 0.5 rad = khoảng 30 độ
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh thành RGB để MediaPipe xử lý
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt và các điểm landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Lấy các điểm landmarks của khuôn mặt
            landmarks = face_landmarks.landmark

            # Kiểm tra xem khuôn mặt có xoay đủ không
            if check_face_rotation(landmarks):
                cv2.putText(frame, "Face rotated enough", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face not rotated enough", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Vẽ các điểm landmarks trên khuôn mặt
            for landmark in landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Hiển thị kết quả
    cv2.imshow('Face Rotation Check', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
