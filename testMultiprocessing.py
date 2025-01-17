import multiprocessing
import time
import cv2
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]
rtsp_urls = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"
# Hàm thực thi cho mỗi process
def worker(n):
    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not cap1.isOpened():
        print("Không thể mở camera")
    else:
        while True:
            ret, frame1 = cap1.read()
            if not ret:
                break
            cv2.imshow(f'Frame {n}', frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap1.release()
    cv2.destroyAllWindows()

# Tạo các process và chạy chúng trên các core khác nhau
if __name__ == "__main__":
    num_processes = 4 # Số lượng process = số lượng core CPU
    processes = []

    # Tạo và khởi động các process
    # worker(5)
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker, args=(i,))
        process.daemon = True
        processes.append(process)
        process.start()

    # Đợi các process hoàn thành
    for process in processes:
        process.join()

    print("All processes finished.")

# from multiprocessing import Pool

# def square(n):
#     return n * n

# if __name__ == "__main__":
#     with Pool(4) as p:  # Sử dụng 4 core CPU
#         result = p.map(square, [1, 2, 3, 4])
#     print(result)
