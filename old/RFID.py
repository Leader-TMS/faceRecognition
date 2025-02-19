# import tkinter as tk

# def handle_rfid_input():
#     rfid_code = entry.get()
#     print(f"RFID Code: {rfid_code}")
#     entry.delete(0, tk.END)

# def delay_handler():
#     root.after(100, handle_rfid_input)

# root = tk.Tk()
# root.title("RFID Scanner")

# label = tk.Label(root, text="Please scan your RFID card:")
# label.pack(pady=10)

# entry = tk.Entry(root, width=30)
# entry.pack(pady=10)

# entry.focus_set()

# entry.bind("<Return>", lambda event: delay_handler())

# root.mainloop()

# from readchar import readkey, key
# input_string = ""

# while True:
#     k = readkey()
    
#     if k not in [key.ENTER, key.BACKSPACE]:
#         input_string += k
#     if k == key.ENTER:
#         print(f"Input received: {input_string}")
#         input_string = ""
    
#     if k == key.BACKSPACE:
#         input_string = input_string[:-1]
#         print(f"Current input: {input_string}")

# def run_rfid_scanner():

print("Đang chờ quét thẻ RFID...")
    
while True:
    # Chờ đợi đầu đọc HID gửi dữ liệu (giống như nhập vào bàn phím)
    rfid_data = input("Quét thẻ RFID: ")
    
    if rfid_data:
        print(f"RFID Data: {rfid_data}")
        break  # Hoặc tiếp tục nếu bạn muốn quét thêm nhiều thẻ