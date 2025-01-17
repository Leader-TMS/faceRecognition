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

import tkinter as tk
from PIL import Image, ImageTk
import cv2

app = tk.Tk()
app.title("Camera Display and Info")
app.geometry("1200x800")
captureFlag = False

cameras = [cv2.VideoCapture(i) for i in range(3)]
labels = []
for i in range(3):
    label = tk.Label(app)
    label.grid(row=0, column=i, padx=10, pady=10)
    labels.append(label)

width, height = 400, 300
for camera in cameras:
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def open_camera(vid, label_widget):
    ret, frame = vid.read()
    if not ret:
        print("Cannot read frame from camera!")
        return
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    if captureFlag:
        cv2.imwrite(f"camera_{1}_image.jpg", frame)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(15, open_camera, vid, label_widget)

def startCapture():
    global captureFlag
    captureFlag = True
    print("Start capturing images.")

def stopCapture():
    global captureFlag
    captureFlag = False
    print("Stopped capturing images.")

def exitProgram():
    print("Exiting program.")
    app.quit()

for i, camera in enumerate(cameras):
    open_camera(camera, labels[i])

labels_input = ["Name:", "Age:", "RFID:"]
entries = []
for i, label_text in enumerate(labels_input):
    label = tk.Label(app, text=label_text)
    label.grid(row=i+3, column=0, padx=10, pady=10)
    entry = tk.Entry(app)
    entry.grid(row=i+3, column=1, padx=10, pady=10)
    entries.append(entry)

submitButton = tk.Button(app, text="Submit Info", command=lambda: print(f"Name: {entries[0].get()}, Age: {entries[1].get()}, RFID: {entries[2].get()}"))
submitButton.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

startButton = tk.Button(app, text="Start Capture", command=startCapture)
startButton.grid(row=2, column=0, padx=10, pady=10)

stopButton = tk.Button(app, text="Stop Capture", command=stopCapture)
stopButton.grid(row=2, column=1, padx=10, pady=10)

exitButton = tk.Button(app, text="Exit", command=exitProgram)
exitButton.grid(row=2, column=2, padx=10, pady=10)

app.mainloop()

