from pathlib import Path
from tkinter import Tk, Canvas, PhotoImage, Label
import cv2
from PIL import Image, ImageTk
import datetime
import random


outputPath = Path(__file__).parent
assetsPath = outputPath / Path(r"/home/minh/Virtual environment/assets/images/facialRecognitionInterface")
dataListGlobal = []

def relativeToAssets(path: str) -> Path:
    return assetsPath / Path(path)

def openCamera(video, labelWidget):
    ret, frame = video.read()
    if not ret:
        print("Cannot read frame from camera!")
        return

    frame = cv2.resize(frame, (750, 439), interpolation=cv2.INTER_LANCZOS4)
    opencvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    capturedImage = Image.fromarray(opencvImage)
    photoImage = ImageTk.PhotoImage(image=capturedImage)

    labelWidget.photoImage = photoImage
    labelWidget.configure(image=photoImage)
    labelWidget.after(25, openCamera, video, labelWidget)

def addNewEntry(code, name, image, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    newEntry = {
        "image": image,
        "code": code,
        "name": name,
        "timestamp": timestamp
    }

    dataListGlobal.insert(0, newEntry)

    if len(dataListGlobal) > 6:
        dataListGlobal.pop()

    renderInfoListFromFixedLayout(canvas, dataListGlobal)

def renderInfoListFromFixedLayout(canvas, dataList, spacing=65):
    positions = []

    for i in range(6):
        yOffset = i * spacing
        positions.append({
            "imgY": 139 + yOffset + 5,
            "titleCodeY": 115 + yOffset + 5,
            "codeY": 115 + yOffset + 5,
            "nameY": 129 + yOffset + 5,
            "titleTimeY": 152 + yOffset + 5,
            "timeY": 152 + yOffset + 5
        })

    if hasattr(canvas, "_infoItems"):
        for itemId in canvas._infoItems:
            canvas.delete(itemId)
        canvas._infoItems.clear()
    else:
        canvas._infoItems = []

    if not hasattr(canvas, "_photoRefs"):
        canvas._photoRefs = []

    for i, entry in enumerate(dataList[:6]):
        pos = positions[i]

        try:
            pilImage = Image.open(relativeToAssets(entry["image"]))
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        pilImage = pilImage.resize((50, 50))
        tkImage = ImageTk.PhotoImage(pilImage)

        imgId = canvas.create_image(828, pos["imgY"], image=tkImage)
        titleCode = canvas.create_text(856, pos["titleCodeY"], anchor="nw", text="Mã NV:", fill="#767676", font=("Amiko SemiBold", 9 * -1))
        codeId = canvas.create_text(892, pos["codeY"], anchor="nw", text=entry["code"], fill="#000000", font=("Amiko Bold", 9 * -1))
        nameId = canvas.create_text(856, pos["nameY"], anchor="nw", text=entry["name"], fill="#000000", font=("Amiko Bold", 9 * -1))
        titleTime = canvas.create_text(856, pos["titleTimeY"], anchor="nw", text="Thời gian:", fill="#767676", font=("Amiko SemiBold", 9 * -1))
        timeId = canvas.create_text(904, pos["timeY"], anchor="nw", text=entry["timestamp"], fill="#000000", font=("Amiko Bold", 9 * -1))

        canvas._photoRefs.append(tkImage)
        canvas._infoItems.extend([imgId, titleCode, codeId, nameId, titleTime, timeId])

def simulateData():
    now = datetime.datetime.now()
    return [
        {
            "image": "image_2.png",
            "code": i + 1,
            "name": f"Người dùng {i + 1}",
            "timestamp": (now - datetime.timedelta(minutes=i * 2)).strftime("%Y-%m-%d %H:%M:%S")
        }
        for i in range(random.randint(1, 10))
    ]

def updateInfoList():
    dataList = simulateData()
    for entry in dataList:
        addNewEntry(entry["code"], entry["name"], entry["image"], entry["timestamp"])
    window.after(5000, updateInfoList)

window = Tk()
window.title("Face Recognition by TMS")
window.geometry("1024x535")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 535,
    width = 1024,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relativeToAssets("image_1.png"))
image_1 = canvas.create_image(
    897.0,
    295.0,
    image=image_image_1
)

canvas.create_text(
    875.0,
    86.0,
    anchor="nw",
    text="08-05-2025",
    fill="#000000",
    font=("Amiko Bold", 12 * -1)
)

labelWidget = Label(window)
labelWidget.place(x=20, y=76)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

openCamera(camera, labelWidget)

image_image_4 = PhotoImage(
    file=relativeToAssets("image_4.png"))
image_4 = canvas.create_image(
    395.0,
    42.0,
    image=image_image_4
)

updateInfoList()
addNewEntry(code="195623", name="Phạm Ngọc Minh", image="image_2.png", timestamp="08:30")

window.resizable(False, False)
window.mainloop()
