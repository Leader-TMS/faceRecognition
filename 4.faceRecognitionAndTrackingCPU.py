
import cv2
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import normalize
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
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
from PIL import Image
import torch
import imagehash
import configparser
import random
import string
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import subprocess
import signal
import time
mtcnn = MTCNN(margin=40, select_largest=False, selection_method='probability', keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.8])
# inceptionModel = InceptionResnetV1(pretrained='vggface2').eval()
# tracedModel = torch.jit.trace(inceptionModel, torch.randn(1, 3, 112, 112))
# torch.set_grad_enabled(False)
# tracedModel.save('inception_resnet_v1_traced.pt')

tracedModel = torch.jit.load('tracedModel/inceptionResnetV1Traced.pt')
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')

# Setup Camera
devs = os.listdir('/dev')
devVideo = [int(dev[-1]) for dev in devs if dev.startswith('video')]
devVideo = sorted(devVideo)[::2]

#Setup mediapipe
mpFaceDetection = mp.solutions.face_detection
mpFaceMesh = mp.solutions.face_mesh

faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.8, model_selection=1)
faceMesh = mpFaceMesh.FaceMesh()
mpDrawing = mp.solutions.drawing_utils

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

# Setup Data
setTrackerName = {}
seeRFID = False
inputRFID = ""
userAreCheckIn = ""
lock = threading.Lock()
color = Color()
targetWFace = 50
checkTimeRecognition = datetime.now()
imageHash = None
trackingIdAssign = None
trackers = {}
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

reading= False
roi_x1, roi_y1, roi_x2, roi_y2 = 250, 250, 600, 350
points = np.array([[625, 280], [890, 280], [885, 625], [630, 635]], dtype=np.int32)
points = points.reshape((-1, 1, 2))
objTypeText = {}
objCheckName = {}
resetCallTTS = datetime.now()
stopThread = False

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

def checkFormatTXT(objCheckName):
    if len(objCheckName) >= 2:
        value = ', '.join(objCheckName.values())
    else:
        value = next(iter(objCheckName.values()), '')
    return value

def waitFullName():
    global resetCallTTS, objCheckName, stopThread
    if stopThread:
        stopThread = False
    while not stopThread:
        if checkTime(resetCallTTS, 0.5):
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

def saveFaceDirection(image, folderName):
    if  image.size > 0:
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
        target_file_name = os.path.join(folderName, f'{current_time}.jpg')
        cv2.imwrite(target_file_name, image)

def faceAuthentication(frame):
    try:
        label = "Unknown"
        employeeCode = None
        hFace, wFace = frame.shape[:2]
        if frame is not None and frame.size > 0:
            wFace = max(int(wFace * 0.4), 45)
            hFace = max(int(hFace * 0.4), 45)

            frame = cv2.resize(frame, (wFace, hFace), interpolation=cv2.INTER_LANCZOS4)

            faceRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faceMtcnn = mtcnn(faceRgb)

            if faceMtcnn is not None and len(faceMtcnn) > 0:
                for _, face in enumerate(faceMtcnn):
                    with torch.no_grad():
                        # embedding = inceptionModel(face.unsqueeze(0)).detach().cpu().numpy().flatten()
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
                        saveFaceDirection(frame, "evidences/invalid")
                        label = "Unknown"
                        employeeCode = None
    except RuntimeError as e:
            textToSpeech(f"Lỗi xác thực khuôn mặt", 1.2, faceAuthError)
            saveFaceDirection(frame, "evidences/error")
            print(f"Warning: {e}. Skipping this face region.")
    finally:
        return label, employeeCode

def getNameFace(frame = None, trackerId = None):
    if frame is not None and frame.size > 0:
        trackerName, employeeCode = faceAuthentication(frame)
        if isinstance(trackerName, str) and trackerName != "Unknown":
            setTrackerName[trackerId]["name"] = trackerName
            setTrackerName[trackerId]["employeeCode"] = employeeCode
            setTrackerName[trackerId]["Known"] = setTrackerName[trackerId]["Known"] + 1
        else:
            setTrackerName[trackerId]["name"] = "Unknown"
            setTrackerName[trackerId]["employeeCode"] = None
            setTrackerName[trackerId]["Unknown"] = setTrackerName[trackerId]["Unknown"] + 1

def scanRFID():
    global inputRFID, seeRFID
    while True:
        k = readkey()
        with lock:
            if k not in [key.ENTER, key.BACKSPACE]:
                inputRFID += k
            if k == key.ENTER:
                employee = getEmployeesByRFID(inputRFID)
                if employee:
                    userAreCheckIn = employee['short_name']
                    employeeCode = employee['employee_code']
                    saveData = saveAttendance("rfid", employeeCode, genUniqueId())
                    if saveData == False:
                        textToSpeech("Lỗi lưu dữ liệu sever", 1.2)
                    else:
                        textToSpeech(f"Xin chào {userAreCheckIn}", 1.2, personName)
                else:
                    textToSpeech("Rờ ép ID chưa được đăng ký", 1.2, rfidNotRegistered)
                inputRFID = ""
                seeRFID = True
            if k == key.BACKSPACE:
                inputRFID = inputRFID[:-1]

def checkTime(savedTime, second = 0.3):
    return round(((datetime.now() - savedTime).total_seconds()), 2) >= second

def findNextLarger(arrTrackerId, trackingIdAssign):
    for i in range(len(arrTrackerId)):
        if arrTrackerId[i] > trackingIdAssign:
            return arrTrackerId[i]
    return arrTrackerId[0]

def delAndFindNextLarger(arrTrackerId, trackingIdAssign):
    if trackingIdAssign in setTrackerName:
        del setTrackerName[trackingIdAssign]
    return findNextLarger(arrTrackerId, trackingIdAssign)


def getImageHash(image):
    if image is not None and image.size > 0:
        pilImage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imageHash = imagehash.phash(pilImage)
        return imageHash
    else:
        return None

def isInsideRectangle(smallX, smallY, smallW, smallH, largeX, largeY, largeW, largeH):
    if smallX >= largeX and smallY >= largeY:
        if smallX + smallW <= largeX + largeW and smallY + smallH <= largeY + largeH:
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

def goodFaceAngle(image, width, height):
    goodAngle = False
    if image is not None and image.size > 0:
        results = faceMesh.process(image)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                leftEye = landmarks.landmark[173]
                rightEye = landmarks.landmark[398]
                nose = landmarks.landmark[4]
                goodAngle = faceAngle(nose, leftEye, rightEye, width, height)
    return goodAngle

def checkLight(image, threshold = 90):
    avgBrightness = 0
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avgBrightness = np.mean(grayImage)
    return avgBrightness >= threshold

def detectionAndTracking(frame):
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(rgbFrame)
    data = {}
    isFace = True
    if results.detections:
        arrFaceTracking = []
        arrNotFaceTracking = []
        smallFace = []

        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw = rgbFrame.shape[:2]
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            faceDetected = rgbFrame[y:y+h, x:x+w]
        
            if faceDetected is not None and faceDetected.size > 0:

                hframe, wframe = faceDetected.shape[:2]
                light = checkLight(frame[y:y+h, x:x+w])
                goodAngle = goodFaceAngle(frame[y:y+h, x:x+w], wframe, hframe)
                blur = checkBlurry(frame[y:y+h, x:x+w])
                
                if  wframe >= 55 and wframe < 65:
                    smallFace.append(0)
                elif wframe >= 65:
                    smallFace.append(1)
                    scaleW = int(w * 1.55)
                    scaleH = int(h * 1.55)
                    scaleX = int(x - (scaleW - w) / 2)
                    scaleY = int(y - (scaleH - h) / 2)

                    if not trackers:
                        trackers[1] = ((x, y, w, h), (scaleX, scaleY, scaleW, scaleH), light, goodAngle, False, blur)
                        arrFaceTracking.append(1)
                    else:
                        for key, (bbox1, bbox2, _, _, _, _) in list(trackers.items()):
                            print(f'speed: {abs(x - bbox1[0]) / 1}')
                            lowSpeed = abs(x - bbox1[0]) / 1 <= 5
                            scaleX2, scaleY2, scaleW2, scaleH2 = bbox2
                            inside = isInsideRectangle(x, y, w, h, scaleX2, scaleY2, scaleW2, scaleH2)
                            if inside:
                                if key in setTrackerName and setTrackerName[key]['timekeeping']:
                                    goodAngle = True
                                trackers[key] = ((x, y, w, h), (scaleX, scaleY, scaleW, scaleH), light, goodAngle, lowSpeed, blur)
                                arrFaceTracking.append(key)
                                break
                            else:
                                arrNotFaceTracking.append(key)
                                newKey = key + 1
                                if newKey not in trackers and newKey > max(trackers.keys()):
                                    trackers[newKey] = ((x, y, w, h), (scaleX, scaleY, scaleW, scaleH), light, goodAngle, lowSpeed, blur)
                                    arrFaceTracking.append(newKey)
                    # cv2.rectangle(frame, (scaleX, scaleY), (scaleX + scaleW, scaleY + scaleH), (0, 255, 0), 2)
        
        if len(smallFace):
            if 1 in smallFace:
                arrFaceTracking = set(arrFaceTracking)
                arrNotFaceTracking = set(arrNotFaceTracking)
                
                if len(arrNotFaceTracking):
                    for num in arrNotFaceTracking:
                        if num not in arrFaceTracking:
                            del trackers[num]

                elif len(arrFaceTracking):
                    for num in list(trackers.keys()):
                        if num not in arrFaceTracking:
                            del trackers[num]
                else:
                    isFace = False

                if isFace:
                    for key, (bbox1, _, light, checkAngle, lowSpeed, blur) in list(trackers.items()):
                        x, y, w, h = bbox1    
                        scaleW = int(w * 1.2)
                        scaleH = int(h * 1.2)
                        scaleX = int(x + (w - scaleW) / 2)
                        scaleY = int(y + (h - scaleH) / 2)
                        data[key] = ((scaleX, scaleY, scaleW, scaleH), light, checkAngle, lowSpeed, blur)
            else:
                textToSpeech("Vui lòng, tiến tới.", 1.2, faceTooSmall)

    return data

def genUniqueId(length=20):
    characters = string.ascii_letters + string.digits
    unique_id = ''.join(random.choices(characters, k=length))
    return unique_id

def checkBlurry(image, threshold=100):
    if image is not None and image.size > 0:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(grayImage, cv2.CV_64F)
        variance = laplacian.var()
        if variance < threshold:
            return True
    return False

def compareAkaze(image1, image2):
    if image1 is not None and image2 is not None and image1.size > 0 and image2.size > 0:
        # hFace1, wFace1 = image1.shape[:2]
        # hFace2, wFace2 = image2.shape[:2]
        # if hFace1 < 65 or wFace1 < 65 or hFace2 < 65 or wFace2 < 65:
        #     return 0
        # else:
        img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        akaze = cv2.AKAZE_create()

        _, des1 = akaze.detectAndCompute(img1, None)
        _, des2 = akaze.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0

        if des1.dtype != des2.dtype:
            des2 = des2.astype(des1.dtype)

        if des1.shape[1] != des2.shape[1]:
            raise ValueError("Descriptors have different number of columns!")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        totalMatches = len(matches)
        if totalMatches == 0:
            return 0

        goodMatches = sum([1 for match in matches if match.distance < 50])
        similarityPercentage = round(goodMatches / totalMatches, 2)
        return 1 if similarityPercentage > 0.6 else 2
    else:
        return 0

def corruptImageDetected(image):
    
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grayFloat = np.float32(grayImg)
    f = np.fft.fft2(grayFloat)
    fshift = np.fft.fftshift(f)

    magnitudeSpectrum = np.log(np.abs(fshift) + 1)

    lowFrequencyEnergy = np.sum(magnitudeSpectrum[:10, :10])
    highFrequencyEnergy = np.sum(magnitudeSpectrum[10:, 10:])

    if lowFrequencyEnergy > highFrequencyEnergy:
        return True
    else:
        return False
    
def textToSpeechSleep(text, speed=1.0):
    tts = gTTS(text=text, lang='vi', slow=False)
    fp =io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio = AudioSegment.from_mp3(fp)

    if speed != 1.0:
        audio = audio.speedup(playback_speed=speed)
    play(audio)

    del fp
    del audio
    del tts
    videoCapture()

def motionBlurCompensation(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    return sharpened

def displayText(frame, x, y, w, light, angle, lowSpeed, blur, color):
    x = x + w
    conditions = [
        (not light, "no light", color.Black),
        (not angle, "look straight", color.Blue),
        (not lowSpeed, "slow down", color.Red),
        (blur, "blur", (167, 80, 167))
    ]
    
    textY = y - 10
    
    for condition, text, rectColor in conditions:
        if condition:
            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            cv2.rectangle(frame, (x, textY - textHeight - 12), (x + textWidth + 10, textY + 5), rectColor, -1)
            
            cv2.putText(frame, text, (x + 4, int(textY - 5)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color.White, 2)
            
            textY += textHeight + 18

def videoCapture():
    global checkTimeRecognition, trackingIdAssign, imageHash, setTrackerName, trackers

    rtsp = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0"
    video = "output_video.avi"
    video2 = "output_video2.avi"
    video3 = "6863054282731021257.mp4"
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # 0.75 ON
    # 0.25 OFF
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE , -4)
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sleepTimer = datetime.now()
    second = 0
    countSecond = 0
    if not cap.isOpened():
        cap.release()
        cv2.destroyAllWindows()
        textToSpeechSleep("Không tìm thấy máy ảnh, Vui lòng kiểm tra lại", 1.2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            textToSpeechSleep("Máy ảnh bị mất tín hiệu, Bắt đầu khởi động lại phần mềm", 1.2)
            break
        else:
            # frame = motionBlurCompensation(frame)
            originalFrame = frame.copy()
            detections = {}
            arrDetection = {}
            if checkLight(frame):
                detections = detectionAndTracking(frame)
            if len(detections):

                sleepTimer = datetime.now() + timedelta(minutes=1)
                for key, (bbox, ligth, angle, lowSpeed, blur) in list(detections.items()):
                    trackerId = key
                    x, y, w, h = bbox

                    w = x + w
                    h = y + h

                    topPoint = int((x + w) / 2)
                    checkPointLine = cv2.pointPolygonTest(points, (topPoint, h), False)
                    checkPointLine = 1
                    if checkPointLine > 0:
                        arrDetection[trackerId] = (bbox, ligth, angle, lowSpeed, blur)

                if len(arrDetection):
                    arrTrackerId = sorted(arrDetection.keys()) if arrDetection is not None else []
                    if trackingIdAssign is None:
                        trackingIdAssign = min(arrTrackerId, default=None)
                        
                    if trackingIdAssign not in arrTrackerId:
                        trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)

                    if trackingIdAssign not in setTrackerName:
                        setTrackerName[trackingIdAssign] = {"name": "Unknown", "employeeCode": None, "Unknown": 0, "Known": 0, "timekeeping": False, "numCheck": 0}
                    
                    if setTrackerName[trackingIdAssign]["numCheck"] >= 4:
                        if setTrackerName[trackingIdAssign]["timekeeping"]:
                            trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                        else:
                            if setTrackerName[trackingIdAssign]["name"] != "Unknown":
                                percentage = round((setTrackerName[trackingIdAssign]['Unknown'] / setTrackerName[trackingIdAssign]['Known']) * 100, 2)
                                if percentage <= 33.3:
                                    setTrackerName[trackingIdAssign]["timekeeping"] = True
                                    (x, y, w, h), _, _, _, _= arrDetection[trackingIdAssign]
                                    updateInfo(setTrackerName[trackingIdAssign]['employeeCode'], originalFrame[y:y+h, x:x+w])
                                    trackingIdAssign = findNextLarger(arrTrackerId, trackingIdAssign)
                                else:
                                    imageHash = None
                                    trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                                    # textToSpeech('Chưa nhận diện được', 1.2, auth)
                            else:
                                imageHash = None
                                trackingIdAssign = delAndFindNextLarger(arrTrackerId, trackingIdAssign)
                                # textToSpeech('Chưa nhận diện được', 1.2, auth)
                    else:
                        isOther = False
                        (x, y, w, h), ligth, angle, lowSpeed, blur = arrDetection[trackingIdAssign]
                        if ligth and angle and lowSpeed:
                            if blur:
                                imgOpt = motionBlurCompensation(originalFrame[y:y+h, x:x+w])
                                blur = checkBlurry(imgOpt)
                                croppedFace = imgOpt
                                arrDetection[trackingIdAssign] = ((x, y, w, h), ligth, angle, lowSpeed, blur)
                                detections[trackingIdAssign] = ((x, y, w, h), ligth, angle, lowSpeed, blur)
                            else:
                                croppedFace = originalFrame[y:y+h, x:x+w]

                            if not blur:
                                if croppedFace is not None and croppedFace.size > 0:
                                    resizeFace = cv2.resize(croppedFace, (croppedFace.shape[1], croppedFace.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                                    corruptImage = corruptImageDetected(resizeFace)

                                    if corruptImage:
                                        # textToSpeech("Ảnh bị mờ", 1.2, imageUnclear)
                                        print("Ảnh bị mờ")
                                    else:
                                        isCompare = compareAkaze(imageHash, resizeFace)
                                        if imageHash is None:
                                            imageHash = resizeFace
                                            isOther = True
                                        else:
                                            if isCompare == 2:
                                                imageHash = resizeFace
                                                isOther = True
                                            elif isCompare == 1 or isCompare == 0:
                                                if setTrackerName[trackingIdAssign]["name"] != "Unknown":
                                                    setTrackerName[trackingIdAssign]["Known"] = setTrackerName[trackingIdAssign]["Known"] + 1
                                                else:
                                                    setTrackerName[trackingIdAssign]["Unknown"] = setTrackerName[trackingIdAssign]["Unknown"] + 1
                                if isOther:
                                    # textToSpeech('Bắt đầu nhận diện', 1.2, auth)
                                    getNameFace(resizeFace, trackingIdAssign)
                        setTrackerName[trackingIdAssign]["numCheck"] += 1

                    checkTimeRecognition = datetime.now()
                for key, (bbox, ligth, angle, lowSpeed, blur) in list(detections.items()):
                    x, y, w, h = bbox
                    if key in setTrackerName:
                        if setTrackerName[key]["timekeeping"]:
                            text = 'Success'
                            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            cv2.rectangle(frame, (x - 15, y - textHeight -15), (x + textWidth + 15, y), (65, 169, 35), -1)
                            cv2.putText(frame, text, (x, int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color.White, 2)
                        else:
                            displayText(frame, x, y, w, ligth, angle, lowSpeed, blur, color)
                            # if not ligth:
                            # if not angle:
                            # if not lowSpeed:
                            # if blur:
                            # text = f"{'look straight' if not angle else ''} {'slow down' if not lowSpeed else ''} {'blur' if blur else ''}"
                            # (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            # cv2.rectangle(frame, (x - 10, y - textHeight - 15), (x + textWidth + 10, y), color.Red, -1)
                            # cv2.putText(frame, text, (x, int(y - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color.White, 2)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), color.Green, 2)
            else:
                trackers = {}
                setTrackerName = {}
                trackingIdAssign = None
                imageHash = None

            if second != datetime.now().second:
                second = datetime.now().second
                if countSecond != 0:
                    fps = countSecond
                countSecond = 1
            else:
                countSecond+=1

            text = f'FPS: {fps} - {trackingIdAssign}'

            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x, y = 30, 30

            cv2.rectangle(frame, (x - 10, y - textHeight - 10), (x + textWidth + 10, y + 10), (167, 80, 167), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.White, 2)
            # if sleepTimer >= datetime.now():
                # subprocess.run(["xset", "dpms", "force", "on"])
                # cv2.namedWindow("ByteTrack", cv2.WINDOW_NORMAL)
                # cv2.setWindowProperty("ByteTrack", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # cv2.imshow("ByteTrack", frame)
            # else:
                # cv2.destroyAllWindows()
                # subprocess.run(["xset", "dpms", "force", "off"])
            # frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow("ByteTrack", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # scanRFIDThread = threading.Thread(target=scanRFID)
    # scanRFIDThread.daemon = True
    # scanRFIDThread.start()

    # videoCaptureThread = threading.Thread(target=videoCapture)
    # videoCaptureThread.start()
    # videoCaptureThread.join()
    videoCapture()
    print("All processes finished.")

# try:
# except Exception as e:
#     print(f"An error occurred: {e}")
#     textToSpeech("Lỗi phát hiện khuôn mặt, Vui lòng liên hệ Admin", 1.2)
# finally:
#     return data
