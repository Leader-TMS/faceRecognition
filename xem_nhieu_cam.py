import cv2
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingCamera']["USER"]
password = config['settingCamera']["PASSWORD"]
ip = config['settingCamera']["IP"]
port = config['settingCamera']["PORT1"]

# rtsp://<User Name>:<Password>@<IP Address>:<Port>/cam/realmonitor?channel=1&subtype=0
rtsp_urls = [
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/", # HT01
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/202/", # HT02
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/302/", # HT03
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/402/", # HT04
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/502/", # Kho H01
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/602/", # Kho HT02
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/702/", # Phong Seal
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/802/", # Rao sau HT
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/902/", # Kho
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1002/", # Tu IT
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1102/", # To cat
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1202/", # Cong 1
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1302/", # Cong 2
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1402/", # Cong 3
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1502/", # Cong 4
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1602/", # Cong 5

    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/202/", # Bon thai 02
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/302/", # Bon thai 01
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/402/", # Cau thang 01
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/502/", # Nha xe
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/602/", # Rao sau BV
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/702/", # Hanh lang sau
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/802/", # Cua bep
    f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/1002/", # Lan can
]

caps = [cv2.VideoCapture(url) for url in rtsp_urls]
width = 720
height = 480
fps = 25

for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Không thể kết nối với Camera {i+1}.")
        continue
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Điều chỉnh exposure_auto (0: tắt tự động, 1: bật tự động)
        cap.set(cv2.CAP_PROP_EXPOSURE, -10)         # Điều chỉnh exposure_absolute (mức độ phơi sáng)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 4246)  # Điều chỉnh white_balance_temperature (Bấm hoặc giảm giá trị)
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 4246)   # Điều chỉnh giá trị màu đỏ trong cân bằng trắng (tùy theo webcam)
        cap.set(cv2.CAP_PROP_FOCUS, 50)            # Điều chỉnh focus_absolute (Điều chỉnh tiêu cự)
        cap.set(cv2.CAP_PROP_SHARPNESS, 150)       # Điều chỉnh sharpness (Độ sắc nét)
        cap.set(cv2.CAP_PROP_ZOOM, 100) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
        cap.set(cv2.CAP_PROP_FPS, fps)


while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Không nhận được frame từ Camera {i+1}.")
            continue
        
        cv2.imshow(f"Camera {i+1}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()