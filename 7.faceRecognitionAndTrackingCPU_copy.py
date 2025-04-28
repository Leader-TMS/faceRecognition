import cv2
import random
import time
import threading
import mediapipe as mp
from collections import OrderedDict
from datetime import datetime
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib
import math
from sklearn.preprocessing import normalize
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
import configparser
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
os.environ['YOLO_VERBOSE'] = 'False'
import threading
from readchar import readkey, key
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play, _play_with_ffplay
import io
from PIL import Image
import imagehash
import string
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import subprocess
import signal
from multiprocessing import Value, Process, Manager, set_start_method, freeze_support
# from faceRecognition import faceRecognition
tracedModel = torch.jit.load('tracedModel/inceptionResnetV1Traced.pt')
torch.set_grad_enabled(False)
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

#Setup mediapipe
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
mpDrawing = mp.solutions.drawing_utils

config = configparser.ConfigParser()
config.read('config.ini')
user = config['settingMyCamera']["USER"]
password = config['settingMyCamera']["PASSWORD"]
ip = config['settingMyCamera']["IP"]
port = config['settingMyCamera']["PORT"]

serverSaveError = config['settingTypeText']["SERVER_SAVE_ERROR"]
personName = config['settingTypeText']["PERSON_NAME"]
faceAuthError = config['settingTypeText']["FACE_AUTH_ERROR"]
rfidNotRegistered = config['settingTypeText']["RFID_NOT_REGISTERED"]
faceTooSmall = config['settingTypeText']["FACE_TOO_SMALL"]
cameraOpenError = config['settingTypeText']["CAMERA_OPEN_ERROR"]
imageUnclear = config['settingTypeText']["IMAGE_UNCLEAR"]
badAngle = config['settingTypeText']["BAD_ANGLE"]
auth = config['settingTypeText']["AUTHENTICATION"]

manager = Manager()
faceJobStates = manager.dict()
trackName = {}
faceSpeedTrackers = {}
objTypeText = {}
objCheckName = {}
resetCallTTS = datetime.now()
stopThread = False
countThread = 3
lightStatus = False
class Color:
    Red = (0, 0, 255)
    Green = (0, 255, 0)
    Blue = (255, 0, 0)
    Yellow = (0, 255, 255)
    Cyan = (255, 255, 0)
    Magenta = (255, 0, 255)
    White = (255, 255, 255)
    Black = (0, 0, 0)
    Gray = (169, 169, 169)
    Orange = (0, 165, 255)
    Pink = (203, 192, 255)
    Purple = (128, 0, 128)
    Brown = (42, 42, 165)
    Violet = (238, 130, 238)
    Lime = (0, 255, 0)
    Olive = (0, 128, 128)
    Teal = (128, 128, 0)
    Silver = (192, 192, 192)
    Gold = (0, 215, 255)

class CentroidTracker:
    def __init__(self):
        self.nextObjectId = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

    def register(self, box):
        self.objects[self.nextObjectId] = box
        self.nextObjectId +=1

    def deregister(self, objectId):
        log(f"[ID {objectId}] Đã biến mất khỏi Frame")
        killJob(objectId)
        del faceJobStates[objectId]
        del self.objects[objectId]

    def scaleBox(self, box, scale=1.55):
        x, y, w, h = box
        newW = int(w * scale)
        newH = int(h * scale)
        newX = int(x - (newW - w) / 2)
        newY = int(y - (newH - h) / 2)
        return (newX, newY, newW, newH)
    
    def isBoxBInsideBoxA(self, boxA, boxB):
        xA, yA, wA, hA = boxA
        xB, yB, wB, hB = boxB
        xA2, yA2 = xA + wA, yA + hA
        xB2, yB2 = xB + wB, yB + hB
        return xB >= xA and yB >= yA and xB2 <= xA2 and yB2 <= yA2
    
    def update(self, rects):
        if len(rects):
            notDisappeared = []
            for index, box in enumerate(rects):
                if not len(self.objects):
                    self.register(box)
                    notDisappeared.append(self.nextObjectId - 1)
                else:
                    matched = False
                    for objectId, trackedBox in list(self.objects.items()):
                        scaledTrackedBox = self.scaleBox(trackedBox)
                        if self.isBoxBInsideBoxA(scaledTrackedBox, box):
                            self.objects[objectId] = box
                            notDisappeared.append(objectId)
                            matched = True
                            break

                    if not matched:
                        self.register(box)
                        notDisappeared.append(self.nextObjectId - 1)

            notDisappeared = set(notDisappeared)
            disappeared = [id for id in self.objects if id not in notDisappeared]

            for id in disappeared:
                self.deregister(id)
        else:
            self.nextObjectId = 0
            self.objects = OrderedDict()

        return self.objects

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

startThread = Process(target=log, args=("Start",))
startThread.daemon = True
startThread.start()

def killJob(objectId):
    print(f"Prepare to kill job object {objectId}...")
    try:
        pid = faceJobStates[objectId]["pid"]
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            print(f"Killed job {pid}")
            
    except ProcessLookupError:
        print("Process kill failed")
        pass

def turnOn():
    global lightStatus
    try:
        if not lightStatus:
            print('ON')
            lightStatus = True
            subprocess.run(
                ['sudo', 'python', 'turnOn.py'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except subprocess.CalledProcessError as e:
        print("Lỗi khi thực thi lệnh:", e)
        print("stdout:", e.stdout.decode())
        print("stderr:", e.stderr.decode())

def turnOff():
    global lightStatus
    try:
        if lightStatus:
            print('OFF')
            lightStatus = False
            subprocess.run(
                ['sudo', 'python', 'turnOff.py'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except subprocess.CalledProcessError as e:
        print("Lỗi khi thực thi lệnh:", e)
        print("stdout:", e.stdout.decode())
        print("stderr:", e.stderr.decode())

def compareAkaze(image1, image2):
    if image1 is not None and image2 is not None and image1.size > 0 and image2.size > 0:

        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        akaze = cv2.AKAZE_create()

        _, des1 = akaze.detectAndCompute(img1, None)
        _, des2 = akaze.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return False

        if des1.dtype != des2.dtype:
            des2 = des2.astype(des1.dtype)

        if des1.shape[1] != des2.shape[1]:
            raise ValueError("Descriptors have different number of columns!")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        totalMatches = len(matches)
        if totalMatches == 0:
            return False

        goodMatches = sum([1 for match in matches if match.distance < 50])
        similarityPercentage = round(goodMatches / totalMatches, 2)
        return similarityPercentage > 0.6
    else:
        return False

def faceRecognition(faceImage):
    try:
        mtcnn = MTCNN(margin=40, select_largest=False, selection_method='probability', keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.8])
        label = None
        employeeCode = None
        hFace, wFace = faceImage.shape[:2]
        if faceImage is not None and faceImage.size > 0:
            wFace = max(int(wFace * 0.4), 45)
            hFace = max(int(hFace * 0.4), 45)

            faceImage = cv2.resize(faceImage, (wFace, hFace), interpolation=cv2.INTER_LANCZOS4)

            faceRgb = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)

            faceMtcnn = mtcnn(faceRgb)

            if faceMtcnn is not None and len(faceMtcnn) > 0:
                for _, face in enumerate(faceMtcnn):
                    with torch.no_grad():
                        embedding = tracedModel(face.unsqueeze(0)).detach().cpu().numpy().flatten()

                    embedding = normalize([embedding])

                    labelIndex = svmModel.predict(embedding)[0]

                    prob = svmModel.predict_proba(embedding)[0]

                    probPercent = round(prob[labelIndex], 2)
                    if probPercent >= 0.75:
                        employeeCode = labelEncoder.inverse_transform([labelIndex])[0]
                        employeeCode = getEmployeesByCode(employeeCode)
                        if employeeCode:
                            label = employeeCode['short_name']
                            employeeCode = employeeCode['employee_code']
                    else:
                        saveFaceDirection(faceImage, "evidences/invalid")
                        label = None
                        employeeCode = None
    except RuntimeError as e:
            textToSpeech(f"Lỗi xác thực khuôn mặt", 1.2, faceAuthError)
            saveFaceDirection(faceImage, "evidences/error")
            print(f"Warning: {e}. Skipping this face region.")
    finally:
        return label, employeeCode

def checkKillProcess():
    objData = list(objTypeText.items())
    if len(objData) >= 2:
        if not any(value is None for _, value in objData):
            if personName in objTypeText:
                maxKey = personName
                maxPid = objTypeText[maxKey]
            else:
                maxKey = max(objTypeText, key=objTypeText.get)
                maxPid = objTypeText[maxKey]
            for key, pid in objData:
                if pid != maxPid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        print("Process kill failed")
                        pass
                    finally:
                        if key in objTypeText:
                            del objTypeText[key]

def generateAndPlayAudio(text, speed, typeId):
    global reading, objTypeText

    reading = True
    tts = gTTS(text=text, lang='vi', slow=False)
    fp =io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio = AudioSegment.from_mp3(fp)

    if speed != 1.0:
        audio = audio.speedup(playback_speed=speed)
    audio = audio + 15
    f = NamedTemporaryFile("w+b", suffix=".wav", delete=False)
    audio.export(f.name, "wav")
    process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "quiet", f.name])
    objTypeText[typeId] = process.pid
    checkKillProcess()
    process.wait()
    if typeId in objTypeText:
        del objTypeText[typeId]
    f.close()
    os.remove(f.name)
        
    # play(audio)

    reading = False
    del fp
    del audio
    del tts

def textToSpeech(text, speed=1.0, typeId = 1):
    global reading
    if typeId not in objTypeText:
        objTypeText[typeId] = None
        thread = threading.Thread(target=generateAndPlayAudio, args=(text, speed, typeId))
        thread.daemon=True
        thread.start()

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def runRecognitionJob(objectId, faceJobStates):
    try:
        faceJobStates[objectId]["pid"] = os.getpid()
        faceImage = name = code = None
        state = faceJobStates[objectId]
        if state["step"] < len(state["list_face"]) and state["list_face"][state["step"]] is not None:
            faceImage = state["list_face"][state["step"]]
            if state["step"] > 0 and compareAkaze(faceImage, state["last_frame"]):
                name = state["result"][0] if len(state["result"]) else None
                code = state["code"]
                log(f"[ID {objectId}] Frame giống trước — dùng lại kết quả {name}")
            else:
                log(f"[ID {objectId}] Frame khác trước — nhận diện mới")
                name, code = faceRecognition(faceImage)
        log(f"[ID {objectId}] Job {state['step']} xong: {name}")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        faceJobStates[objectId]["result"].append(name)
        faceJobStates[objectId]["code"] = code
        faceJobStates[objectId]["step"] += 1
        faceJobStates[objectId]["last_frame"] = faceImage


def displayText(frame, x, y, w, light, smallFace, angle, lowSpeed, blur, color):
    x = x + w
    conditions = [
        (not light, "no light", color.Black),
        (smallFace, "small face", (71, 173, 253)),
        (not angle, "look straight", color.Blue),
        (not lowSpeed, "slow down", color.Red),
        (blur, "blur", (167, 80, 167))
    ]

    textY = y - 10

    for condition, text, rectColor in conditions:
        if condition:
            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, textY - textHeight - 12), (x + textWidth + 10, textY + 5), rectColor, -1)
            cv2.putText(frame, text, (x + 4, int(textY - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.6, color.White, 2)
            textY += textHeight + 18

def checkLight(image, threshold = 90):
    avgBrightness = 0
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avgBrightness = np.mean(grayImage)
    return avgBrightness >= threshold

def checkBlurry(image, threshold=100):
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(grayImage, cv2.CV_64F)
        variance = laplacian.var()
        if variance < threshold:
            return True
    return False

def faceAngle(nose, leftEye, rightEye, fWidth, fHeight):

    noseX, noseY = int(nose.x * fWidth), int(nose.y * fHeight)
    leftEyeX, leftEyeY = int(leftEye.x * fWidth), int(leftEye.y * fHeight)
    rightEyeX, rightEyeY = int(rightEye.x * fWidth), int(rightEye.y * fHeight)
    
    vector1 = np.array([leftEyeX - noseX, leftEyeY - noseY])

    vector2 = np.array([rightEyeX - noseX, rightEyeY - noseY])

    dotProduct = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    cosAngle = round(dotProduct / (magnitude1 * magnitude2), 2)
    return cosAngle <= 0.6

def goodFaceAngle(image):
    goodAngle = False
    height, width = image.shape[:2]
    if image is not None and image.size > 0:
        results = faceMesh.process(image)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                leftEye = landmarks.landmark[173]
                rightEye = landmarks.landmark[398]
                nose = landmarks.landmark[4]
                goodAngle = faceAngle(nose, leftEye, rightEye, width, height)
    return goodAngle

def motionBlurCompensation(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def updateInfo(employeeCode, image):
    global resetCallTTS, stopThread
    fullName = getEmployeesByCode(employeeCode)["short_name"]
    if  fullName:
        if employeeCode not in objCheckName:
            stopThread = True
            resetCallTTS = datetime.now()
            objCheckName[employeeCode] = fullName

        thread = threading.Thread(target=waitFullName)
        thread.daemon=True
        thread.start()
        saveFaceDirection(image, f"evidences/valid/{fullName}")
        if not saveAttendance("face", employeeCode, genUniqueId()):
            textToSpeech("Lỗi lưu dữ liệu server", 1.2)
    else:
        textToSpeech("Mã nhân viên chưa được đăng ký trong hệ thống.", 1.2, 8)
        saveFaceDirection(image, "evidences/invalid")

def genUniqueId(length=20):
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(random.choices(characters, k=length))
    return unique_id

def waitFullName():
    global resetCallTTS, objCheckName, stopThread
    if stopThread:
        stopThread = False
    while not stopThread:
        if checkTime(resetCallTTS, 0.3):
            if len(objCheckName):
                listName = checkFormatTXT(objCheckName)
                if listName !=  '':
                    print(f"Xin chào {listName} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
                    textToSpeech(f"Xin chào {listName}", 1.2, personName)
            objCheckName = {}
            resetCallTTS = datetime.now()
            stopThread = False
            break
        time.sleep(0.1)

def checkFormatTXT(objCheckName):
    if len(objCheckName) >= 2:
        value = ', '.join(objCheckName.values())
    else:
        value = next(iter(objCheckName.values()), '')
    return value

def checkTime(savedTime, second = 0.3):
    return round(((datetime.now() - savedTime).total_seconds()), 2) >= second

def textToSpeechSleep(text, speed=1.0):
    tts = gTTS(text=text, lang='vi', slow=False)
    fp =io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio = AudioSegment.from_mp3(fp)

    if speed != 1.0:
        audio = audio.speedup(playback_speed=speed)
    # play(audio)

    del fp
    del audio
    del tts
    videoCapture()


def videoCapture():
    global faceJobStates, trackName, faceSpeedTrackers, startThread
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.8, model_selection=0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE , -4)
    tracker = CentroidTracker()
    frameCount = 0
    prevTime = time.time()
    fps = 0
    sleepTimer = datetime.now()
    minObjectId = None
    if not cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        textToSpeechSleep("Không tìm thấy máy ảnh, Vui lòng kiểm tra lại", 1.2)
    while True:
        ret, frame = cap.read()
        originalImage = frame.copy()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            textToSpeechSleep("Máy ảnh bị mất tín hiệu, Bắt đầu khởi động lại phần mềm", 1.2)
            break
        else:
            frameCount += 1
            currentTime = time.time()
            if currentTime - prevTime >= 1:
                fps = frameCount
                frameCount = 0
                prevTime = currentTime

            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(frameRgb)

            rects = []
            if results.detections:
                sleepTimer = datetime.now() + timedelta(minutes=1)
                for detection in results.detections:
                    score = detection.score[0]
                    if score >= 0.75:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x = int(bboxC.xmin * w)
                        y = int(bboxC.ymin * h)
                        width = int(bboxC.width * w)
                        height = int(bboxC.height * h)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), Color.White, 2)
                        rects.append((x, y, width, height))

            objects = tracker.update(rects)
            if len(objects):
                for objectId, (x, y, w, h) in objects.items():
                    if objectId not in faceJobStates:
                        faceJobStates[objectId] = manager.dict({
                            "step": 0,
                            "result": manager.list(),
                            "code": None,
                            "last_frame": None,
                            "success": False,
                            "list_face": manager.list(),
                            "pid": None
                        })
                    state = faceJobStates[objectId]
                    filteredIds = [objId for objId, data in faceJobStates.items() if data['success'] == False]
                    minObjectId = min(filteredIds) if filteredIds else None
                    cv2.putText(frame, f'objectId: {objectId}', (x - 50, int(y - 50)), cv2.FONT_HERSHEY_DUPLEX, 0.8, Color.Red, 2)
                    
                    if objectId == minObjectId:
                        faceCrop = originalImage[y:y+h, x:x+w]
                        _, faceWidth = faceCrop.shape[:2]
                        light = checkLight(faceCrop)
                        if light:
                            if  faceWidth >= 55 and faceWidth < 65:
                                displayText(frame, x, y, w, light, True, True, True, False, Color)
                            elif faceWidth >= 65:
                                frameHeight, frameWidth = frame.shape[:2]
                                centerXPixel = x + w // 2
                                centerYPixel = y + h // 2
                                centerXNorm = centerXPixel / frameWidth
                                centerYNorm = centerYPixel / frameHeight

                                currentSpeedTime = time.time()
                                centerNorm = (centerXNorm, centerYNorm)
                                centerPixel = (centerXPixel, centerYPixel)

                                normSpeed = 0
                                pixelSpeed = 0

                                if objectId not in faceSpeedTrackers:
                                    faceSpeedTrackers[objectId] = {
                                        "prev_norm": centerNorm,
                                        "prev_pixel": centerPixel,
                                        "time": currentSpeedTime
                                    }
                                else:
                                    prevData = faceSpeedTrackers[objectId]
                                    deltaTime = currentSpeedTime - prevData["time"]

                                    if deltaTime > 0:
                                        normDist = math.hypot(centerNorm[0] - prevData["prev_norm"][0],
                                                            centerNorm[1] - prevData["prev_norm"][1])
                                        normSpeed = round(normDist / deltaTime, 2)

                                        pixelDist = math.hypot(
                                            centerPixel[0] - prevData["prev_pixel"][0],
                                            centerPixel[1] - prevData["prev_pixel"][1]
                                        )
                                        pixelSpeed = round(pixelDist / deltaTime, 2)

                                        text = f"N:{normSpeed}/s | P:{pixelSpeed}px/s"
                                        cv2.putText(frame, text, (x, y + h + 20),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, Color.Orange, 2)

                                    faceSpeedTrackers[objectId] = {
                                        "prev_norm": centerNorm,
                                        "prev_pixel": centerPixel,
                                        "time": currentSpeedTime
                                    }
                                
                                blur = checkBlurry(faceCrop)
                                angle = goodFaceAngle(faceCrop)
                                lowSpeed = normSpeed < 0.3
                                if state["step"] < countThread and not startThread.is_alive():
                                    if angle and lowSpeed:
                                        if blur:
                                            imgOpt = motionBlurCompensation(faceCrop)
                                            blur = checkBlurry(imgOpt)
                                            faceCrop = imgOpt
                                        if not blur:
                                            if len(state["list_face"]) < countThread:
                                                faceJobStates[objectId]["list_face"].append(faceCrop.copy())
                                            startThread = Process(target=runRecognitionJob, args=(objectId, faceJobStates,))
                                            startThread.daemon = True
                                            faceJobStates[objectId]["pid"] = startThread.pid
                                            startThread.start()
                                        else:
                                            displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)
                                    else:
                                        displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)

                                elif state["step"] < countThread and startThread.is_alive():
                                    if len(state["list_face"]) < countThread:
                                        if angle and lowSpeed:
                                            if blur:
                                                imgOpt = motionBlurCompensation(faceCrop)
                                                blur = checkBlurry(imgOpt)
                                                faceCrop = imgOpt
                                            if not blur:
                                                faceJobStates[objectId]["list_face"].append(faceCrop.copy())

                                elif state["step"] >= countThread:
                                    names = state["result"]
                                    if len(names) == countThread and names.count(None) == 0 and len(set(names)) == 1:
                                        trackName[objectId] = names[0]
                                        if not state["success"]:
                                            faceJobStates[objectId]["success"] = True
                                            killJob(objectId)
                                            updateInfo(state["code"], faceCrop)
                                            log(f"[ID {objectId}] Thành công: {names[0]}")
                                    else:
                                        log(f"[ID {objectId}] → reset lại")
                                        faceJobStates[objectId] = manager.dict({
                                            "step": 0,
                                            "result": manager.list(),
                                            "code": None,
                                            "last_frame": None,
                                            "success": False,
                                            "list_face": manager.list(),
                                            "pid": None
                                        })
                        else:
                            turnOn()
                    name = trackName.get(objectId, "Checking..." if state["step"] > 0 else "New")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if(state["success"]):
                        text = 'Success'
                        (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x - 15, y - textHeight -15), (x + textWidth + 15, y), (65, 169, 35), -1)
                        cv2.putText(frame, text, (x, int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, Color.White, 2)
                    else:
                        cv2.putText(frame, f"{name} (ID {objectId})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        displayText(frame, x, y, w, light, False, True, True, False, Color)
            else:
                turnOff()
                faceJobStates = {}
                trackName = {}
                faceSpeedTrackers = {}

            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)
            if sleepTimer >= datetime.now():
                #subprocess.run(["xset", "dpms", "force", "on"])
                cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.namedWindow("Face Tracking + Recognition", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Face Tracking + Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Face Tracking + Recognition", frame)
            else:
                cv2.destroyAllWindows()
                turnOff()
                #subprocess.run(["xset", "dpms", "force", "off"])
            # cv2.putText(frame, f"FPS: {fps} - minObjectId: {minObjectId}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow("Face Tracking + Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videoCapture()
    turnOff()