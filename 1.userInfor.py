import tkinter as tk
from tkinter import ttk, Tk, Canvas, Entry, Text, Button, PhotoImage, Label, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
from datetime import datetime
from HeadPoseEstimation import cameraForward, resetData
import os
from dataProcessing import getEmployeesByCode

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
        opencvImage = cameraForward(frame, None, entry1.get())
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
    if captureFlag:
        stopCapture()
        # startButton.config(text="Bắt đầu (chụp hình)")
    else:
        startCapture()  # Bắt đầu chụp ảnh
        # startButton.config(text="Kết thúc")

def startCapture():
    global captureFlag, buttonImage2
    employeeCode = entry1.get()

    if not employeeCode:
        messagebox.showwarning("Lỗi đầu vào", "Vui lòng nhập mã nhân viên!")
        return
    
    if getEmployeesByCode(employeeCode) is None:
        messagebox.showwarning("Thông báo", "Không tìm thấy thông tin, vui lòng đăng ký qua trang: https://dangkythongtin.vn/login")
        return
    
    captureFlag = True
    buttonImage2 = PhotoImage(file=relativeToAssets("button_3.png"))
    button2.config(image=buttonImage2)
    print(f"create employeeCode: {employeeCode}")
    folderPath = os.path.join("dataset", employeeCode)
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

def stopCapture():
    global captureFlag, buttonImage2
    resetData()
    buttonImage2 = PhotoImage(file=relativeToAssets("button_2.png"))
    button2.config(image=buttonImage2)
    captureFlag = False
    print("Stopped capturing images.")

def exitProgram():
    print("Exiting program.")
    window.quit()

def checkEmployeesByCode(event):
    employeeCode = entry1.get()
    if employeeCode:
        employeeCode = getEmployeesByCode(employeeCode)
        if employeeCode is None:
            messagebox.showwarning("Thông báo", "Không tìm thấy thông tin, vui lòng đăng ký thông qua trang: https://dangkythongtin.vn/login")
            return
        else:
            canvas.itemconfig(nameId, text=employeeCode['full_name'], font=("Amiko-Bold", 14 * -1))

            imgBbox = canvas.bbox(image2)
            imgWidth = imgBbox[2] - imgBbox[0]

            textBbox = canvas.bbox(nameId)
            textWidth = textBbox[2] - textBbox[0]

            while textWidth > imgWidth:
                current_font = canvas.itemcget(nameId, "font")
                current_size = int(current_font.split()[1])
                if current_size < 0:
                    new_size = current_size + 1
                else:
                    new_size = current_size - 1
                
                canvas.itemconfig(nameId, font=("Amiko-Bold", new_size))
                
                textBbox = canvas.bbox(nameId)
                textWidth = textBbox[2] - textBbox[0]

            newX = imgBbox[0] + (imgWidth - textWidth) / 2

            canvas.coords(nameId, newX, 156.0)

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
    766.5,
    237.5,
    image=entryImage1
)
entry1 = Entry(
    bd=0,
    bg="#F6F7F9",
    fg="#000716",
    highlightthickness=0,
)
entry1.place(
    x=662.0,
    y=227.0,
    width=209.0,
    height=19.0
)
entry1.focus_set()
entry1.bind('<Return>', checkEmployeesByCode)

imageImage4 = PhotoImage(
    file=relativeToAssets("image_4.png"))
image4 = canvas.create_image(
    767.0,
    231.0,
    image=imageImage4
)

imageImage5 = PhotoImage(
    file=relativeToAssets("image_5.png"))
image5 = canvas.create_image(
    766.0,
    441.0,
    image=imageImage5
)

imageImage6 = PhotoImage(
    file=relativeToAssets("image_6.png"))
image6 = canvas.create_image(
    766.0,
    115.0,
    image=imageImage6
)

nameId = canvas.create_text(
    695.0, 156.0, anchor="nw", text="Nhập mã nhân viên", fill="#575757", font=("Amiko-Bold", 14 * -1)
)

imageImage7 = PhotoImage(
    file=relativeToAssets("image_7.png"))
image7 = canvas.create_image(
    771.0,
    31.0,
    image=imageImage7
)

imageImage8 = PhotoImage(
    file=relativeToAssets("image_8.png"))
image8 = canvas.create_image(
    766.0,
    284.0,
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