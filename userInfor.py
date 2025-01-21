# import cv2
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# import threading
# import time
# import configparser

# config = configparser.ConfigParser()
# config.read('config.ini')
# user = config['settingCamera']["USER"]
# password = config['settingCamera']["PASSWORD"]
# ip = config['settingCamera']["IP"]
# port = config['settingCamera']["PORT1"]
# rtsp_urls = f"rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/102/"

# root = tk.Tk()
# root.title("Camera Display and Info")
# root.geometry("1200x800")

# captureFlag = False
# inputName = ""
# inputAge = ""
# inputRfid = ""

# def getCameraFeed(cameraId, label):
#     cap = cv2.VideoCapture(rtsp_urls)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(frame)
#         img = img.resize((400, 300))
#         imgTk = ImageTk.PhotoImage(img)

#         label.imgTk = imgTk
#         label.config(image=imgTk)
        
#         if captureFlag:
#             cv2.imwrite(f"camera_{cameraId}_image.jpg", frame)

#     cap.release()
#     cv2.destroyAllWindows()

# def startCapture():
#     global captureFlag
#     captureFlag = True
#     print("Start capturing images.")

# def stopCapture():
#     global captureFlag
#     captureFlag = False
#     print("Stopped capturing images.")

# def submitInfo():
#     global inputName, inputAge, inputRfid
#     inputName = nameEntry.get()
#     inputAge = ageEntry.get()
#     inputRfid = rfidEntry.get()
#     print(f"Name: {inputName}, Age: {inputAge}, RFID: {inputRfid}")

# def exitProgram():
#     print("Exiting program.")
#     root.quit()

# cameraLabels = []
# for i in range(3):
#     cameraLabel = tk.Label(root)
#     cameraLabel.grid(row=0, column=i, padx=10, pady=10)
#     cameraLabels.append(cameraLabel)

# startButton = tk.Button(root, text="Start Capture", command=startCapture)
# startButton.grid(row=1, column=0, padx=10, pady=10)

# stopButton = tk.Button(root, text="Stop Capture", command=stopCapture)
# stopButton.grid(row=1, column=1, padx=10, pady=10)

# exitButton = tk.Button(root, text="Exit", command=exitProgram)
# exitButton.grid(row=1, column=2, padx=10, pady=10)

# nameLabel = tk.Label(root, text="Name:")
# nameLabel.grid(row=2, column=0, padx=10, pady=10)
# nameEntry = tk.Entry(root)
# nameEntry.grid(row=2, column=1, padx=10, pady=10)

# ageLabel = tk.Label(root, text="Age:")
# ageLabel.grid(row=3, column=0, padx=10, pady=10)
# ageEntry = tk.Entry(root)
# ageEntry.grid(row=3, column=1, padx=10, pady=10)

# rfidLabel = tk.Label(root, text="RFID:")
# rfidLabel.grid(row=4, column=0, padx=10, pady=10)
# rfidEntry = tk.Entry(root)
# rfidEntry.grid(row=4, column=1, padx=10, pady=10)

# submitButton = tk.Button(root, text="Submit Info", command=submitInfo)
# submitButton.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# threads = []
# for i in range(3):
#     t = threading.Thread(target=getCameraFeed, args=(i, cameraLabels[i]))
#     t.daemon = True
#     threads.append(t)
#     t.start()

# root.mainloop()

# import tkinter as tk
# import cv2 
# from PIL import Image, ImageTk 
  
# # Define a video capture object 
# vid = cv2.VideoCapture(0) 
  
# # Declare the width and height in variables 
# width, height = 800, 600
  
# # Set the width and height 
# vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
# vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
  
# # Create a GUI app 
# app = tk.Tk() 
  
# # Bind the app with Escape keyboard to 
# # quit app whenever pressed 
# app.bind('<Escape>', lambda e: app.quit()) 
  
# # Create a label and display it on app 
# label_widget = tk.Label(app) 
# label_widget.grid(row=0, column=0, padx=10, pady=10)
# label_widget.pack() 
  
# # Create a function to open camera and 
# # display it in the label_widget on app 
  
  
# def open_camera(): 
  
#     # Capture the video frame by frame 
#     _, frame = vid.read() 
  
#     # Convert image from one color space to other 
#     opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
  
#     # Capture the latest frame and transform to image 
#     captured_image = Image.fromarray(opencv_image) 
  
#     # Convert captured image to photoimage 
#     photo_image = ImageTk.PhotoImage(image=captured_image) 
  
#     # Displaying photoimage in the label 
#     label_widget.photo_image = photo_image 
  
#     # Configure image in the label 
#     label_widget.configure(image=photo_image) 
  
#     # Repeat the same process after every 10 seconds 
#     label_widget.after(15,open_camera) 
  
  
# # Create a button to open the camera in GUI app 
# button1 = tk.Button(app, text="Open Camera", command=open_camera) 
# button1.pack() 
  
# # Create an infinite loop for displaying app on screen 
# app.mainloop() 
#-------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, Tk, Canvas, Entry, Text, Button, PhotoImage, Label, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
from datetime import datetime
from HeadPoseEstimation import cameraForward
import os

OUTPUT_PATH = os.path.dirname(__file__)
ASSETS_PATH = os.path.join(OUTPUT_PATH, "assets", "images")

def relativeToAssets(path: str) -> str:
    return os.path.join(ASSETS_PATH, path)

def openCamera(vid, labelWidget):
    ret, frame = vid.read()
    if not ret:
        print("Cannot read frame from camera!")
        return
    
    if captureFlag:
        opencvImage = cameraForward(frame, None, entry2.get())
        opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_RGB2BGR)
    else:
        frame = cv2.flip(frame, 1)
        opencvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    capturedImage = Image.fromarray(opencvImage)
    photoImage = ImageTk.PhotoImage(image=capturedImage)

    labelWidget.photoImage = photoImage
    labelWidget.configure(image=photoImage)
    labelWidget.after(25, openCamera, vid, labelWidget)

def toggleCapture():
    global captureFlag
    if captureFlag:
        stopCapture()  # Dừng chụp ảnh
        # startButton.config(text="Bắt đầu (chụp hình)")  # Đổi tên nút
    else:
        startCapture()  # Bắt đầu chụp ảnh
        # startButton.config(text="Kết thúc")  # Đổi tên nút

def startCapture():
    global captureFlag
    folderName = entry2.get()

    if not folderName:
        messagebox.showwarning("Input Error", "Vui lòng nhập mã nhân viên!")
        return
    
    captureFlag = True
    folderPath = os.path.join("dataset", folderName)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def stopCapture():
    global captureFlag
    captureFlag = False
    print("Stopped capturing images.")

def exitProgram():
    print("Exiting program.")
    window.quit()

window = Tk()
window.geometry("894x480")
window.configure(bg="#FFFFFF")
window.title("Camera Display and Info")
captureFlag = False

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=480,
    width=894,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
imageImage1 = PhotoImage(
    file=relativeToAssets("image_1.png"))
image1 = canvas.create_image(
    767.0,
    240.0,
    image=imageImage1
)

imageImage2 = PhotoImage(
    file=relativeToAssets("image_2.png"))
image2 = canvas.create_image(
    767.0,
    230.0,
    image=imageImage2
)

labelWidget = Label(window)
labelWidget.place(x=0, y=0)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

openCamera(camera, labelWidget)

buttonImage1 = PhotoImage(
    file=relativeToAssets("button_1.png"))
button1 = Button(
    image=buttonImage1,
    borderwidth=0,
    highlightthickness=0,
    command=exitProgram,
    relief="flat"
)
button1.place(
    x=657.0,
    y=348.0,
    width=220.0,
    height=30.0
)

buttonImage2 = PhotoImage(
    file=relativeToAssets("button_2.png"))
button2 = Button(
    image=buttonImage2,
    borderwidth=0,
    highlightthickness=0,
    command=toggleCapture,
    relief="flat"
)
button2.place(
    x=657.0,
    y=308.0,
    width=220.0,
    height=30.0
)

entryImage1 = PhotoImage(
    file=relativeToAssets("entry_1.png"))
entryBg1 = canvas.create_image(
    764.5,
    275.5,
    image=entryImage1
)
entry1 = Entry(
    bd=0,
    bg="#F6F7F9",
    fg="#000716",
    highlightthickness=0
)
entry1.place(
    x=660.0,
    y=265.0,
    width=209.0,
    height=19.0
)

imageImage4 = PhotoImage(
    file=relativeToAssets("image_4.png"))
image4 = canvas.create_image(
    766.0,
    269.0,
    image=imageImage4
)

imageImage5 = PhotoImage(
    file=relativeToAssets("image_5.png"))
image5 = canvas.create_image(
    767.0,
    213.0,
    image=imageImage5
)

entryImage2 = PhotoImage(
    file=relativeToAssets("entry_2.png"))
entryBg2 = canvas.create_image(
    765.5,
    219.5,
    image=entryImage2
)
entry2 = Entry(
    bd=0,
    bg="#F6F7F9",
    fg="#000716",
    highlightthickness=0
)
entry2.place(
    x=661.0,
    y=209.0,
    width=209.0,
    height=19.0
)

imageImage6 = PhotoImage(
    file=relativeToAssets("image_6.png"))
image6 = canvas.create_image(
    766.0,
    441.0,
    image=imageImage6
)

canvas.create_text(
    705.0,
    156.0,
    anchor="nw",
    text="Phạm Ngọc Minh ",
    fill="#575757",
    font=("Amiko-Bold", 14 * -1)
)

imageImage7 = PhotoImage(
    file=relativeToAssets("image_7.png"))
image7 = canvas.create_image(
    766.0,
    115.0,
    image=imageImage7
)

imageImage8 = PhotoImage(
    file=relativeToAssets("image_8.png"))
image8 = canvas.create_image(
    771.0,
    31.0,
    image=imageImage8
)

# labelsInput = ["Name:", "Gender:", "RFID:"]
# entries = []
# for i, labelText in enumerate(labelsInput):        
#     label = Label(window, text=labelText).grid(row=i, column=1, padx=10, pady=10)
#     if labelText == "Gender:":
#         entry = ttk.Combobox(window, values=["Male", "Female", "Other"], state="readonly").grid(row=i + 1, column=1, padx=10, pady=10)
#     else:
#         entry = Entry(window).grid(row=i + 1, column=1, padx=10, pady=10, sticky= tk.S)
#     entries.append(entry)
    
# submitButton = Button(window, text="Xác nhận", command=lambda: print(f"Name: {entries[0].get()}, RFID: {entries[1].get()}"))
# submitButton.grid(row=6, column=1, padx=10, pady=10)

# startButton = Button(window, text="Bắt đầu (chụp hình)", command=toggleCapture)
# startButton.grid(row=4, column=1, padx=10, pady=10)

# exitButton = Button(window, text="Exit", command=exitProgram)
# exitButton.grid(row=4, column=1, columnspan=4, padx=10, pady=10)

window.resizable(False, False)
window.mainloop()