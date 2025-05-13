# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# # img1 = np.full((300, 300, 3), (255, 0, 0), dtype=np.uint8)
# # img2 = np.full((300, 300, 3), (0, 255, 0), dtype=np.uint8)
# img1 = cv2.imread('evidences/valid/Minh Thông/2025-05-07_08-18-00_892.jpg')
# img2 = cv2.imread('evidences/valid/Quốc Anh/2025-05-07_08-10-45_334.jpg')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frameHeight, frameWidth = frame.shape[:2]
#     # newWidth = int(frameWidth * 0.7)
#     # newHeight = int(frameHeight * 0.7)

#     # resizedFrame = cv2.resize(frame, (frameWidth, frameHeight), interpolation=cv2.INTER_LANCZOS4)

#     side = min(frameHeight // 2, 120)
#     img1Resized = cv2.resize(img1, (side, side))
#     img2Resized = cv2.resize(img2, (side, side))

#     usedHeight = side * 2
#     remainingHeight = frameHeight - usedHeight
#     bottomPadding = np.full((remainingHeight, side, 3), 200, dtype=np.uint8)

#     sideHeader = np.vstack((img1Resized, img2Resized, bottomPadding))

#     finalFrame = np.hstack((frame, sideHeader))

#     cv2.imshow('Frame', finalFrame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def drawTextVietnamese(img, text, position, fontPath='fonts/Roboto-Regular.ttf', fontSize=20, color=(0, 0, 0)):
    imgPIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imgPIL)
    font = ImageFont.truetype(fontPath, fontSize)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(imgPIL), cv2.COLOR_RGB2BGR)

def shortenText(text, maxLength):
    return text if len(text) <= maxLength else text[:maxLength - 3] + '...'

def createInfoBlock(name, time, count, width, height, fontPath):
    block = np.full((height, width, 3), 230, dtype=np.uint8)

    block = drawTextVietnamese(block, f'Họ & tên: {shortenText(name, 30)}', (10, 5), fontPath=fontPath, fontSize=16)
    block = drawTextVietnamese(block, f'Thời gian: {time}', (10, 30), fontPath=fontPath, fontSize=16)
    block = drawTextVietnamese(block, f'Số lần: {count}', (10, 55), fontPath=fontPath, fontSize=16)

    return block

fontPath = 'fonts/Roboto-Regular.ttf'

img1 = cv2.imread('evidences/valid/Minh Thông/2025-05-07_08-18-00_892.jpg')
img2 = cv2.imread('evidences/valid/Quốc Anh/2025-05-07_08-10-45_334.jpg')

if img1 is None or img2 is None:
    print("Không load được ảnh.")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frameHeight, frameWidth = frame.shape[:2]
    newWidth = int(frameWidth * 0.7)
    newHeight = int(frameHeight * 0.7)

    resizedFrame = cv2.resize(frame, (newWidth, newHeight), interpolation=cv2.INTER_LANCZOS4)

    side = 80
    infoWidth = 200

    img1Resized = cv2.resize(img1, (side, side))
    img2Resized = cv2.resize(img2, (side, side))

    info1 = createInfoBlock("Trần Minh Thông", "12:34", 5, infoWidth, side, fontPath)
    info2 = createInfoBlock("Nguyễn Quốc Anh", "12:35", 3, infoWidth, side, fontPath)

    block1 = np.hstack((img1Resized, info1))
    block2 = np.hstack((img2Resized, info2))

    # Separator giữa các cụm (ảnh + thông tin)
    separator = np.full((2, block1.shape[1], 3), (100, 100, 100), dtype=np.uint8)

    # Ghép các cụm thành cột
    sideHeader = np.vstack((block1, separator, block2))

    usedHeight = sideHeader.shape[0]
    remainingHeight = newHeight - usedHeight
    if remainingHeight > 0:
        bottomPadding = np.full((remainingHeight, sideHeader.shape[1], 3), 200, dtype=np.uint8)
        sideHeader = np.vstack((sideHeader, bottomPadding))

    finalFrame = np.hstack((resizedFrame, sideHeader))

    cv2.imshow('FrameWithGroupedBlocks', finalFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
