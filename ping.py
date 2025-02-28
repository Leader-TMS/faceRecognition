import subprocess
import time

def ping_ip(ip_address):
    """Ping địa chỉ IP và trả về True nếu có thể kết nối."""
    try:
        # Sử dụng lệnh ping của hệ thống (tương thích cả Windows/Linux)
        response = subprocess.run(["ping", "-c", "1", ip_address], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Kiểm tra mã trả về
        if response.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Lỗi khi ping: {e}")
        return False

def check_ping_range(start_ip, end_ip, interval=2):
    """Ping dải IP từ start_ip đến end_ip và in kết quả."""
    start = int(start_ip.split(".")[-1])  # Lấy phần cuối của IP bắt đầu
    end = int(end_ip.split(".")[-1])      # Lấy phần cuối của IP kết thúc
    base_ip = ".".join(start_ip.split(".")[:-1])  # Phần chung của dải IP (192.168.40)

    for i in range(start, end + 1):
        ip = f"{base_ip}.{i}"
        # print(f"Đang kiểm tra: {ip}")
        if ping_ip(ip):
            print(f"{ip} có thể kết nối.")
        # else:
        #     print(f"{ip} không thể kết nối.")
        
        # Chờ một khoảng thời gian trước khi ping IP tiếp theo
        time.sleep(interval)

# Ví dụ sử dụng:
start_ip = "192.168.40.10"
end_ip = "192.168.40.100"
check_ping_range(start_ip, end_ip, interval=1)  # Điều chỉnh thời gian chờ (interval) nếu cần
