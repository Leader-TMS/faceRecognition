import cv2
import configparser
from threading import Thread

# Đọc cấu hình từ file INI
config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]

rtsp_url = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"
rtsp_url1 = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/202/"

# Hàm xử lý video cho từng luồng
def process_video(video_url, window_name):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_url}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Lỗi khi đọc video từ {video_url}")
            break
        cv2.imshow(window_name, frame)

        # Nếu nhấn 'q', thoát khỏi vòng lặp video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)  # Đảm bảo đóng cửa sổ khi luồng kết thúc

# Hàm khởi chạy tất cả các luồng video
def start_threads():
    # Tạo các luồng video riêng biệt cho từng camera
    video_thread_1 = Thread(target=process_video, args=(rtsp_url, "Video 1"))
    video_thread_2 = Thread(target=process_video, args=(rtsp_url1, "Video 2"))
    
    # Khởi động các luồng
    video_thread_1.start()
    video_thread_2.start()

    # Chờ các luồng kết thúc (trong trường hợp bạn muốn dừng ứng dụng sau khi các video kết thúc)
    video_thread_1.join()
    video_thread_2.join()

if __name__ == "__main__":
    start_threads()
