import cv2
import torch
from facenet_pytorch import MTCNN
import os
import numpy as np

# Khởi tạo MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Mở webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Không thể mở webcam")

print("Nhấn SPACE để chụp ảnh và kiểm tra các mức nén...")
print("Nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Hiển thị khung webcam
    cv2.imshow('Webcam (Nhấn SPACE để chụp)', frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        original = frame.copy()
        break

cap.release()
cv2.destroyAllWindows()

# Tạo thư mục lưu ảnh nếu cần
os.makedirs('compressed_tests', exist_ok=True)

# Lặp qua các mức nén JPEG
for quality in range(100, 10, -20):  # 100, 80, 60, 40, 20
    # Nén ảnh vào bộ nhớ (không cần lưu ra ổ cứng)
    result, encimg = cv2.imencode('.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    img_npy = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    # Phát hiện khuôn mặt
    img_rgb = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    # Vẽ khung quanh khuôn mặt (nếu có)
    if boxes is not None:
        for box in boxes:
            (x1, y1, x2, y2) = map(int, box)
            cv2.rectangle(img_npy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ghi thông tin lên ảnh
    num_faces = 0 if boxes is None else len(boxes)
    text = f"JPEG Quality: {quality} - Faces: {num_faces}"
    cv2.putText(img_npy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị ảnh
    cv2.imshow(f'Quality {quality}', img_npy)

# Đợi người dùng đóng cửa sổ
print("Nhấn bất kỳ phím nào trong cửa sổ ảnh để thoát...")
cv2.waitKey(0)
cv2.destroyAllWindows()
