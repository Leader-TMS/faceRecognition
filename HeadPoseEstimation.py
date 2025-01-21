import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
import datetime


mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDrawing = mp.solutions.drawing_utils
drawingSpec = mpDrawing.DrawingSpec(thickness=1, circle_radius=1)
# looking = {"right": [], "right_up":[], "right_down": [], "left": [], "left_up":[], "left_down":[], "down": [], "up": []}
looking = {"right": [], "left": [], "down": [], "up": []}

def checkAndSaveFace(value, direction, faceFilename, faceImg):
    value = round(value,1)
    if value not in looking[direction] and faceImg.size > 0:
        looking[direction].append(value)

        cv2.imwrite(faceFilename, faceImg)

def increaseAndRound(value):
    if value < 0:
        value -= 0.3
    else:
        value += 0.3
    return round(value,1)

def drawEllipseForProgress(frame, xFace, yFace, wFace, hFace, looking):
    center = ((xFace + wFace) // 2, int(((yFace + hFace) // 2) * 0.95))
    radius = min((wFace - xFace) + ((wFace - xFace) * 0.4), (hFace - yFace) + ((hFace - yFace) * 0.4)) // 2

    circumference = 2 * math.pi * radius
    
    anglePerRay = 360 / (circumference // (radius * 0.1 * 0.8))
    numRays = int(circumference / (radius * 0.1 * 0.8))

    minRays = {
        "right": 40,
        "left": 40,
        "down": 40,
        "up": 40,
        # "right_up": 40,
        # "right_down": 40,
        # "left_up": 40,
        # "left_down": 40
    }

    isDone = 0
    for direction in minRays:
        if len(looking[direction]) >= minRays[direction]:
            isDone +=1

    if isDone < len(minRays):
        for i in range(numRays):
            angle = i * anglePerRay
            xStart = int(center[0] + radius * math.cos(math.radians(angle)))
            yStart = int(center[1] + radius * math.sin(math.radians(angle)))

            rayLength = (wFace - xFace) * 0.1
            xEnd = int(xStart + rayLength * math.cos(math.radians(angle)))
            yEnd = int(yStart + rayLength * math.sin(math.radians(angle)))
            
            # if (angle >= 0 and angle < 22.5) or (angle >= 337.5 and angle < 360):
            #     direction = "right"
            # elif angle >= 22.5 and angle < 67.5:
            #     direction = "right_down"
            # elif angle >= 67.5 and angle < 112.5:
            #     direction = "down"
            # elif angle >= 112.5 and angle < 157.5:
            #     direction = "left_down"
            # elif angle >= 157.5 and angle < 202.5:
            #     direction = "left"
            # elif angle >= 202.5 and angle < 247.5:
            #     direction = "left_up"
            # elif angle >= 247.5 and angle < 292.5:
            #     direction = "up"
            # elif angle >= 292.5 and angle < 337.5:
            #     direction = "right_up"

            print(f"angle: {angle}")
            if (angle >= 0 and angle < 45) or (angle >= 315 and angle < 360):
                direction = "right"
            elif angle >= 45 and angle < 135:
                direction = "down"
            elif angle >= 135 and angle < 225:
                direction = "left"
            elif angle >= 225 and angle < 315:
                direction = "up"

            if direction in looking and len(looking[direction]) > 0:
                progress = len(looking[direction]) / minRays[direction]
                progress = min(progress, 1.0)
                if progress >= 0.65 and progress <= 0.95:
                    progress = 0.65
                blue = 0
                green = int(progress * 255)
                red = int((1 - progress) * 255)

                cv2.line(frame, (xStart, yStart), (xEnd, yEnd), (red, green, blue), 2)
            else:
                cv2.line(frame, (xStart, yStart), (xEnd, yEnd), (206, 242, 189), 2)
        return True
    else:
        cv2.putText(frame, "Done", (center[0] - 40, yFace - int(hFace * 0.2)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return False
    
def cameraForward(frame, windowName = None, folderName = None):
    # start = time.time()

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    frame.flags.writeable = False

    results = faceMesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    imgH, imgW, imgC = frame.shape

    face3d = []
    face2d = []

    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:

            now = datetime.datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
            xFace = int(min([lm.x for lm in faceLandmarks.landmark]) * imgW)
            yFace = int(min([lm.y for lm in faceLandmarks.landmark]) * imgH)
            wFace = int(max([lm.x for lm in faceLandmarks.landmark]) * imgW)
            hFace = int(max([lm.y for lm in faceLandmarks.landmark]) * imgH)
            checkWFace = wFace - xFace
            checkHFace = hFace - yFace
            targetSize = 160

            if checkWFace < targetSize:
                plus = (targetSize - checkWFace) // 2
                xFace = xFace - plus
                wFace = wFace + plus
            elif checkWFace > targetSize:
                minus = (checkWFace - targetSize) // 2
                xFace = xFace + minus
                wFace = wFace - minus
            if checkHFace < targetSize:
                plus = (targetSize - checkHFace) // 2
                yFace = yFace - plus
                hFace = hFace + plus
            elif checkHFace > targetSize:
                minus = (checkHFace - targetSize) // 2
                yFace = yFace + minus
                hFace = hFace - minus

            faceImg = frame[yFace:hFace, xFace:wFace].copy()
            h, w = faceImg.shape[:2]
            if h != 160 and faceImg.size > 0 or w != 160 and faceImg.size > 0:
                faceImgResized = cv2.resize(faceImg, (160, 160))
                print(f"Resized face image dimensions: {faceImgResized.shape[0]}x{faceImgResized.shape[1]}")
            else:
                faceImgResized = faceImg 

            if folderName is None:
                faceFilename = f"stored-faces/face_{current_time}.jpg"
            else:
                faceFilename = f"dataset/{folderName}/face_{current_time}.jpg"

            for idx, lm in enumerate(faceLandmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose2d = (lm.x * imgW, lm.y * imgH)
                        nose3d = (lm.x * imgW, lm.y * imgH, lm.z * 3000)
                    x, y = int(lm.x * imgW), int(lm.y * imgH)
                    face2d.append([x, y])
                    face3d.append([x, y, lm.z])

            face2d = np.array(face2d, dtype=np.float64)
            face3d = np.array(face3d, dtype=np.float64)

            # focalLength = 1 * imgW

            camMatrix = np.array([[imgW, 0, imgH / 2],
                                  [0, imgW, imgW / 2],
                                  [0, 0, 1]])

            distMatrix = np.zeros((4, 1), dtype=np.float64)

            success, rotVec, transVec = cv2.solvePnP(face3d, face2d, camMatrix, distMatrix)

            rmat, jac = cv2.Rodrigues(rotVec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            checkSave = drawEllipseForProgress(frame, xFace, yFace, wFace, hFace, looking)
            text = "Forward"
            if checkSave:
                if y < -3.5:
                    checkAndSaveFace(y, "left", faceFilename, faceImgResized)
                    text = "Looking Left"
                    # checkMore = False
                    # if y >= -15:
                        # if x < -5:
                        #     checkMore = True
                        #     checkAndSaveFace(x, "left_down", faceFilename, faceImgResized)
                        #     text += " Down"
                        # elif x > 6.5:
                        #     checkMore = True
                        #     checkAndSaveFace(x, "left_up", faceFilename, faceImgResized)
                        #     text += " Up"
                        # if not checkMore:
                            # checkAndSaveFace(y, "left", faceFilename, faceImgResized)
                    # else:
                        # if x < 3:
                        #     checkMore = True
                        #     checkAndSaveFace(x, "left_down", faceFilename, faceImgResized)
                        #     text += " Down"
                        # elif x > 10:
                        #     checkMore = True
                        #     checkAndSaveFace(x, "left_up", faceFilename, faceImgResized)
                        #     text += " Up"
                        # if not checkMore:
                            # checkAndSaveFace(y, "left", faceFilename, faceImgResized)
                elif y > 3.5:
                    checkAndSaveFace(y, "right", faceFilename, faceImgResized)
                    text = "Looking Right"
                    # checkMore = False
                    # if y < 10:
                    #     if x < -5:
                    #         checkMore = True
                    #         checkAndSaveFace(x, "right_down", faceFilename, faceImgResized)
                    #         text += " Down"
                    #     elif x > 6.5:
                    #         checkMore = True
                    #         checkAndSaveFace(x, "right_up", faceFilename, faceImgResized)
                    #         text += " Up"
                    #     if not checkMore:
                    #         checkAndSaveFace(y, "right", faceFilename, faceImgResized)
                    # else:
                    #     if x < 3:
                    #         checkMore = True
                    #         checkAndSaveFace(x, "right_down", faceFilename, faceImgResized)
                    #         text += " Down"
                    #     elif x > 10:
                    #         checkMore = True
                    #         checkAndSaveFace(x, "right_up", faceFilename, faceImgResized)
                    #         text += " Up"
                    #     if not checkMore:
                    #         checkAndSaveFace(y, "right", faceFilename, faceImgResized)
                elif x < -2:
                    checkAndSaveFace(x, "down", faceFilename, faceImgResized)
                    text = "Looking Down"
                elif x > 5:
                    checkAndSaveFace(x, "up", faceFilename, faceImgResized)
                    text = "Looking Up"
                else:
                    text = "Forward"

            nose3dProjection, jacobian = cv2.projectPoints(nose3d, rotVec, transVec, camMatrix, distMatrix)
            p1 = (int(nose2d[0]), int(nose2d[1]))
            p2 = (int(nose2d[0] + y * 10), int(nose2d[1] - x * 10))

            cv2.line(frame, p1, p2, (255, 0, 0), 3)
            (textWidth, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            xText = ((xFace + wFace) // 2) - (textWidth // 2)
            yText = int(hFace * 1.2)

            if checkSave:
                cv2.putText(frame, text, (xText, yText), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "x: " + str(np.round(x, 2)), (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "y: " + str(np.round(y, 2)), (450, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "z: " + str(np.round(z, 2)), (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # cv2.putText(frame, f"Up: {len(looking['up'])}", (500, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Down: {len(looking['down'])}", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Right: {len(looking['right'])}", (500, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Left: {len(looking['left'])}", (500, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # cv2.putText(frame, f"Left Up: {len(looking['left_up'])}", (500, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Left Down: {len(looking['left_down'])}", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Right Up: {len(looking['right_up'])}", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # cv2.putText(frame, f"Right Down: {len(looking['right_down'])}", (500, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # end = time.time()
        # totalTime = end - start
        # fps = 1 / totalTime
        # cv2.putText(frame, f'FPS: {int(fps)}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # mpDrawing.draw_landmarks(
        #     image=frame,
        #     landmark_list=faceLandmarks,
        #     connections=mpFaceMesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=drawingSpec,
        #     connection_drawing_spec=drawingSpec)

    if not windowName:
        return frame
    
    cv2.imshow(windowName, frame)

if __name__ == "__main__":
    devs = os.listdir('/dev')
    cameraIds = [int(dev[-1]) for dev in devs 
                if dev.startswith('video')]
    cameraIds = sorted(cameraIds)[::2]
    caps = [cv2.VideoCapture(cameraId) for cameraId in cameraIds]

    # 1024x768 = 640x480
    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                cameraForward(frame, f"Camera {i + 1} - Face Recognition")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()

    cv2.destroyAllWindows()
