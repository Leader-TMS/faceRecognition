import cv2
import mediapipe as mp
import os

# Khởi tạo thư mục lưu ảnh khuôn mặt
SAVE_DIR = "saved_faces"
os.makedirs(SAVE_DIR, exist_ok=True)
face_id = 0  # Đếm số lượng ảnh đã lưu

# Khởi tạo các đối tượng của Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Mở webcam

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                # Lấy bounding box tương đối
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                # Đảm bảo tọa độ không bị âm
                x, y = max(0, x), max(0, y)

                # Tính tỉ lệ scale để chiều dài ít nhất 200px
                target_size = 200
                scale = max(target_size / w, target_size / h)

                new_w = int(w * scale)
                new_h = int(h * scale)

                # Tính lại vị trí để crop với kích thước đã scale
                cx = x + w // 2
                cy = y + h // 2

                x1 = max(cx - new_w // 2, 0)
                y1 = max(cy - new_h // 2, 0)
                x2 = min(cx + new_w // 2, iw)
                y2 = min(cy + new_h // 2, ih)

                # Crop và lưu khuôn mặt
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_filename = os.path.join(SAVE_DIR, f"face_{face_id}.jpg")
                    cv2.imwrite(face_filename, face_crop)
                    print(f"[OK] Đã lưu {face_filename}")
                    face_id += 1

                # Vẽ bounding box trên khung hình gốc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
