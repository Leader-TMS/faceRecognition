import face_recognition

# Tải ảnh
image1 = face_recognition.load_image_file("dataset/001/face_2025-02-21_16-29-20_306.jpg")
image2 = face_recognition.load_image_file("temporaryFace/tracking_id_0.jpg")

# Tìm kiếm khuôn mặt trong các ảnh
face_encoding1 = face_recognition.face_encodings(image1)[0]
face_encoding2 = face_recognition.face_encodings(image2)[0]

# So sánh độ tương đồng
results = face_recognition.compare_faces([face_encoding1], face_encoding2)

# In kết quả
if results[0]:
    print("Hai khuôn mặt giống nhau.")
else:
    print("Hai khuôn mặt khác nhau.")
