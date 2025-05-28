import os
import logging
os.environ['YOLO_VERBOSE'] = 'False'
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
import cv2
import random
import time
import threading
from datetime import datetime
import numpy as np
import torch
import joblib
from dataProcessing import getEmployeesByCode, getEmployeesByRFID, saveAttendance
import configparser
import joblib
import numpy as np
from datetime import datetime, timedelta
import threading
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play, _play_with_ffplay
import io
import string
from tempfile import NamedTemporaryFile
import subprocess
import signal
from multiprocessing import Process, Manager
import socket
from facesController import CentroidTracker, CameraReader
import requests
from PIL import ImageFont, ImageDraw, Image, ImageTk
from tkinter import Tk, Canvas, PhotoImage, Label
import faiss
from insightface.model_zoo import get_model
from insightface.utils import face_align

torch.set_grad_enabled(False)
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')
index = faiss.read_index('index/faiss.index')
ids = joblib.load('index/ids.pkl')

detModel = get_model('models/scrfd_500m_bnkps.onnx')
detModel.prepare(ctx_id=-1, input_size=(320, 320))

recModel = get_model('models/w600k_mbf.onnx')
recModel.prepare(ctx_id=-1)

faissIndex = faiss.read_index('indexWithInsight/faiss.index')
labelList = joblib.load('indexWithInsight/labels.pkl')

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
urlGetEmbedding = config['settingApi']["URL_GET_EMBEDDING"]

manager = Manager()
faceJobStates = manager.dict()
objTypeText = {}
objCheckName = {}
resetCallTTS = datetime.now()
stopThread = False
countThread = 3
lightStatus = False
frameCount = 0
prevTime = time.time()
fps = 0
sleepTimer = datetime.now()
minObjectId = None
worked = False
session = requests.Session() 
dataList = []
dataListLock = threading.Lock()
# NamedTemporaryFile
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

def log(msg, saveToFile=False):
    logMessage = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    if saveToFile:
        os.makedirs("logs", exist_ok=True)
        currentDate = datetime.now().strftime('%Y-%m-%d')
        logFilename = os.path.join("logs", f"log_{currentDate}.txt")
        with open(logFilename, 'a', encoding='utf-8') as f:
            f.write(logMessage)
    print(logMessage)


startThread = Process(target=log, args=("Start",))
startThread.daemon = True
startThread.start()

def getIndexCam():
    try:
        devs = os.listdir('/dev')
        devVideo = [int(dev[5:]) for dev in devs if dev.startswith('video') and dev[5:].isdigit()]
        devVideo = sorted(devVideo)
        arrCam = []
        for index in devVideo:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    arrCam.append(index)
                    # return index
            else:
                cap.release()
        return arrCam
    except Exception as e:
        log(f"getIndexCam: {e}", True)
        return arrCam

def killJob(objectId):
    log(f"Prepare to kill job object {objectId}...")
    try:
        jobState = faceJobStates.get(objectId)
        if jobState:
            pid = jobState.get("pid")
        else:
            pid = None
        if objectId in faceJobStates:
            del faceJobStates[objectId]
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
            log(f"Killed job {pid}")
            
    except ProcessLookupError:
        log("Process kill failed", True)
        pass

def turnOn():
    global lightStatus
    try:
        if not lightStatus:
            log('ON')
            lightStatus = True
            subprocess.run(
                ['sudo', 'python', 'turnOn.py'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except subprocess.CalledProcessError as e:
        log(f"Error executing command: {e}", True)
        log(f"Stdout: {e.stdout.decode()}", True)
        log(f"Stderr: {e.stdout.decode()}", True)

def turnOff():
    global lightStatus
    try:
        if lightStatus:
            log('OFF')
            lightStatus = False
            subprocess.run(
                ['sudo', 'python', 'turnOff.py'],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except subprocess.CalledProcessError as e:
        log(f"Error executing command: {e}", True)
        log(f"Stdout: {e.stdout.decode()}", True)
        log(f"Stderr: {e.stdout.decode()}", True)

def compareAkaze(image1, image2):
    try:
        if image1 is not None and image2 is not None and image1.size > 0 and image2.size > 0:
            img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            akaze = cv2.AKAZE_create()
            _, des1 = akaze.detectAndCompute(img1, None)
            _, des2 = akaze.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                log("Descriptor(s) could not be computed.")
                return False

            if des1.dtype != des2.dtype:
                des2 = des2.astype(des1.dtype)

            if des1.shape[1] != des2.shape[1]:
                raise ValueError("Descriptors have different number of columns.")

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            totalMatches = len(matches)
            if totalMatches == 0:
                log("No matches found.")
                return False

            goodMatches = sum([1 for match in matches if match.distance < 50])
            similarityPercentage = round(goodMatches / totalMatches, 2)
            return similarityPercentage > 0.6
        else:
            log("One or both images are empty or None.")
            return False

    except cv2.error as e:
        log(f"OpenCV error occurred: {e}", True)
        return False
    except ValueError as e:
        log(f"Value error: {e}", True)
        return False
    except Exception as e:
        log(f"Unexpected error: {e}", True)
        return False

def faceRecognition(faceImage):
    try:
        label = None
        employeeCode = None
        hFace, wFace = faceImage.shape[:2]
        embedding = None
        if faceImage is not None and faceImage.size > 0:

            embedding = recModel.get_feat(faceImage)
            if embedding is not None:
                emb = embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(emb)
                D, I = faissIndex.search(emb, 1)
                sim = D[0][0]
                if sim >= 0.6:
                    employeeCode = labelList[I[0][0]]
                    employeeCode = getEmployeesByCode(employeeCode)
                    if employeeCode:
                        label = employeeCode['short_name']
                        employeeCode = employeeCode['employee_code']
                elif sim >= 0.5:
                    # employeeCode = labelList[I[0][0]]
                    # employeeCode = getEmployeesByCode(employeeCode)
                    # if employeeCode:
                    #     label = employeeCode['short_name']
                    #     employeeCode = employeeCode['employee_code']
                    #     saveFaceDirection(faceImage, f"evidences/almostSimilar/{employeeCode}_{label}")
                    label = None
                    employeeCode = None
                else:
                    saveFaceDirection(faceImage, "evidences/invalid")
                    label = None
                    employeeCode = None
            else:
                label = None
                employeeCode = None
    except RuntimeError as e:
            textToSpeech(f"Lỗi xác thực khuôn mặt", 1.2, faceAuthError)
            saveFaceDirection(faceImage, "evidences/error")
            log(f"Warning: {e}. Skipping this face region.", True)
    finally:
        return label, employeeCode

def checkKillProcess():
    try:
        objData = list(objTypeText.items())

        if len(objData) >= 2 and not any(value is None for _, value in objData):
            if 'personName' in globals() and personName in objTypeText:
                maxKey = personName
                maxPid = objTypeText[maxKey]
            else:
                if not all(isinstance(pid, int) for pid in objTypeText.values()):
                    log("Invalid PID types found in objTypeText.")
                    return
                maxKey = max(objTypeText, key=objTypeText.get)
                maxPid = objTypeText[maxKey]

            for key, pid in objData:
                if pid != maxPid:
                    try:
                        if not isinstance(pid, int):
                            raise TypeError(f"PID for {key} is not an integer.")
                        os.kill(pid, signal.SIGKILL)
                        log(f"Process {pid} ({key}) killed successfully.")
                    except ProcessLookupError:
                        log(f"Process {pid} ({key}) does not exist.")
                    except PermissionError:
                        log(f"No permission to kill process {pid} ({key}).")
                    except TypeError as e:
                        log(f"Invalid PID type: {e}")
                    except Exception as e:
                        log(f"Unexpected error killing process {pid} ({key}): {e}")
                    else:
                        try:
                            del objTypeText[key]
                        except KeyError:
                            log(f"Key {key} was already removed from objTypeText.")
        else:
            log("Not enough valid process data to perform kill operation.")

    except NameError as e:
        log(f"Missing variable: {e}", True)
    except Exception as e:
        log(f"Unexpected error in checkKillProcess(): {e}", True)

def generateAndPlayAudio(text, speed, typeId):
    global reading, objTypeText

    try:
        reading = True
        tts = gTTS(text=text, lang='vi', slow=False)
        
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        audio = AudioSegment.from_mp3(fp)

        if speed != 1.0:
            audio = audio.speedup(playback_speed=speed)

        f = NamedTemporaryFile("w+b", suffix=".wav", delete=False)
        audio.export(f.name, "wav")
        
        try:
            process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "quiet", f.name]
            )
            objTypeText[typeId] = process.pid
            checkKillProcess()
            process.wait()

        except subprocess.SubprocessError as e:
            log(f"Subprocess error occurred: {e}", True)

        finally:
            if typeId in objTypeText:
                del objTypeText[typeId]
            f.close()
            os.remove(f.name)

        del fp
        del audio
        del tts

    except Exception as e:
        log(f"An error occurred while generating and playing audio: {e}", True)
    
    finally:
        reading = False

def textToSpeech(text, speed=1.0, typeId=1):
    global reading

    try:
        if typeId not in objTypeText:
            objTypeText[typeId] = None
            thread = threading.Thread(target=generateAndPlayAudio, args=(text, speed, typeId))
            thread.daemon = True
            thread.start()
        else:
            log(f"Process for typeId {typeId} is already running.")
    
    except (TypeError, ValueError) as e:
        log(f"Error in textToSpeech for typeId {typeId}: {e}", True)
    
    except Exception as e:
        log(f"Unexpected error in textToSpeech for typeId {typeId}: {e}", True)

def saveFaceDirection(image, folderName):
    try:
        if image is not None and image.size > 0:
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            else:
                if not os.access(folderName, os.W_OK):
                    raise PermissionError(f"Cannot write to directory: {folderName}")

            now = datetime.now()
            currentTime = now.strftime("%Y-%m-%d_%H-%M-%S") + f"_{now.microsecond // 1000}"
            targetFileName = os.path.join(folderName, f'{currentTime}.jpg')

            success = cv2.imwrite(targetFileName, image)
            if not success:
                log(f"Failed to save the image to {targetFileName}")
                return None
            else:
                log(f"Image saved successfully to {targetFileName}")
                return targetFileName
        else:
            log("Invalid image provided. Image is either None or has zero size.")
            return None
    
    except PermissionError as e:
        log(f"Permission error: {e}", True)
        return None
    except cv2.error as e:
        log(f"OpenCV error: {e}", True)
        return None
    except Exception as e:
        log(f"An unexpected error occurred while saving the image: {e}", True)
        return None

def runRecognitionJob(objectId, faceJobStates):
    try:
        faceJobStates[objectId]["pid"] = os.getpid()
        name = code = None
        state = faceJobStates[objectId]

        for getImage in state["list_face"]:
            try:
                imageBytes = np.frombuffer(getImage, dtype=np.uint8)
                faceImage = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
                test = 0
                if faceImage is not None:
                    try:
                        if state["step"] > 0 and compareAkaze(faceImage, state["last_frame"]) and test:
                            name = state["result"][0] if len(state["result"]) else None
                            code = state["code"]
                            log(f"[ID {objectId}] Frame giống trước — dùng lại kết quả {name}")
                        else:
                            log(f"[ID {objectId}] Frame khác trước — nhận diện mới")
                            name, code = faceRecognition(faceImage)
                    except Exception as e:
                        log(f"[ID {objectId}] Error in face comparison or recognition: {e}", True)
                        continue

                if name is None:
                    killJob(objectId)
                    break
                
                faceJobStates[objectId]["result"].append(name)
                faceJobStates[objectId]["code"] = code
                faceJobStates[objectId]["step"] += 1
                faceJobStates[objectId]["last_frame"] = faceImage
                log(f"[ID {objectId}] Job {state['step']} xong: {name}")

            except Exception as e:
                log(f"[ID {objectId}] Error in processing frame: {e}", True)
                continue

    except Exception as e:
        log(f"Error in runRecognitionJob for objectId {objectId}: {e}", True)
        killJob(objectId)

def checkLight(image, threshold=90):
    try:
        avgBrightness = 0
        if image is not None and image.size > 0:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avgBrightness = np.mean(grayImage)
        return avgBrightness >= threshold
    except cv2.error as e:
        log(f"OpenCV error in checkLight: {e}", True)
        return False
    except Exception as e:
        log(f"Unexpected error in checkLight: {e}", True)
        return False

def wrapText(text, maxLength=21, maxTotalLength=41):

    if len(text) <= maxLength:
        return (text, False)
    
    spaceIndex = text.rfind(" ", 0, maxLength + 1)
    if spaceIndex == -1:
        firstLine = text[:maxLength]
        secondLine = text[maxLength:]
    else:
        firstLine = text[:spaceIndex]
        secondLine = text[spaceIndex + 1:]
    
    if len(firstLine) + len(secondLine) > maxTotalLength:
        remain = maxTotalLength - len(firstLine) - 3
        secondLine = secondLine[:remain] + "..."
    
    return (firstLine + "\n" + secondLine, True)
    
def updateInfo(employeeCode, image):
    global resetCallTTS, stopThread
    try:
        if not len(objCheckName):
            isRunSpeech = True
        else:
            isRunSpeech = False

        employeeData = getEmployeesByCode(employeeCode)
        if employeeData:
            fullName = employeeData["full_name"]
            shortName = employeeData["short_name"]

            if not fullName or not shortName:
                textToSpeech("Thiếu thông tin quan trọng: Họ tên hoặc tên ngắn chưa có.", 1.2)
                return

            if employeeCode not in objCheckName:
                stopThread = True
                resetCallTTS = datetime.now()
                objCheckName[employeeCode] = shortName

            if isRunSpeech:
                thread = threading.Thread(target=waitFullName)
                thread.daemon = True
                thread.start()

            saveFaceDirection(image, f"evidences/valid/{shortName}")

            thread = threading.Thread(
                target=addNewEntry,
                kwargs={
                    'code': employeeCode,
                    'name': wrapText(fullName),
                    'image': f"avatar/{employeeCode}.jpg"
                }
            )
            thread.daemon = True
            thread.start()
            saveAttendance("face", employeeCode, genUniqueId())
            
        else:
            textToSpeech("Mã nhân viên chưa được đăng ký trong hệ thống.", 1.2, 8)
            saveFaceDirection(image, "evidences/invalid")
    except Exception as e:
        log(f"Error in updateInfo for employeeCode {employeeCode}: {e}", True)
        textToSpeech("Đã xảy ra lỗi trong quá trình cập nhật thông tin.", 1.2)

def genUniqueId(length=20):
    try:
        characters = string.ascii_letters + string.digits
        unique_id = ''.join(random.choices(characters, k=length))
        return unique_id
    except Exception as e:
        log(f"Error in genUniqueId: {e}", True)
        return None

def waitFullName(seenCodes=None, isFirstCall=True):
    global objCheckName

    if seenCodes is None:
        seenCodes = set()
    try:
        currentCount = len(objCheckName)
        newItems = {
            code: name for code, name in objCheckName.items()
            if code not in seenCodes
        }
        if newItems:
            text = ", ".join(newItems.values())
            if text:
                seenCodes.update(newItems.keys())
                if isFirstCall:
                    generateAndPlayAudio(f"Xin chào {text}", 1.2, personName)
                else:
                    generateAndPlayAudio(f"{text}", 1.2, personName)

            if len(objCheckName) > currentCount:
                waitFullName(seenCodes, isFirstCall=False)
            else:
                objCheckName = {} 
        else:
            objCheckName = {}  
    except Exception as e:
        objCheckName = {}  
        print(f"[waitFullName] Error: {e}")

def textToSpeechSleep(inputData, speed=1.0, restart=True):
    try:
        isFile = False
        if os.path.isfile(inputData):
            isFile = True
            audio = AudioSegment.from_file(inputData)
        else:
            tts = gTTS(text=inputData, lang='vi', slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            audio = AudioSegment.from_mp3(fp)

        if speed != 1.0:
            audio = audio.speedup(playback_speed=speed)
        play(audio)

        if not isFile:
            del fp
            del audio
            del tts
        # videoCapture()
    except Exception as e:
        log(f"Error in textToSpeechSleep: {e}", True)
        # videoCapture()
    finally:
        if restart:
            openCam()

def checkInternet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def tileImages(images, gridSize=None, imageSize=(360, 480)):
    if gridSize is None:
        gridCols = int(np.ceil(np.sqrt(len(images))))
        gridRows = int(np.ceil(len(images) / gridCols))
    else:
        gridRows, gridCols = gridSize

    blankImage = np.zeros((imageSize[0], imageSize[1], 3), dtype=np.uint8)
    normalized = []

    for img in images:
        resized = cv2.resize(img, (imageSize[1], imageSize[0]))
        normalized.append(resized)

    while len(normalized) < gridRows * gridCols:
        normalized.append(blankImage.copy())

    rows = []
    for i in range(0, gridRows * gridCols, gridCols):
        row = cv2.hconcat(normalized[i:i + gridCols])
        rows.append(row)

    return cv2.vconcat(rows)

def resizeWithPadding(image, targetWidth=605, targetHeight=454, color=(0, 0, 0)):
    h, w = image.shape[:2]
    scale = min(targetWidth / w, targetHeight / h)
    newW, newH = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (newW, newH), interpolation=cv2.INTER_LANCZOS4)

    deltaW = targetWidth - newW
    deltaH = targetHeight - newH
    top, bottom = deltaH // 2, deltaH - deltaH // 2
    left, right = deltaW // 2, deltaW - deltaW // 2

    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def handleImage(labelWidget, tracker):
    global faceJobStates, startThread, frameCount, prevTime, fps, sleepTimer, minObjectId, worked, textSizeCache
    try:
        frames = []
        for i, cam in enumerate(cameras):
            camFrame = cam.getFrame()
            if camFrame is None or not cam.getStatus():
                camFrame = cv2.putText(
                    np.zeros((360, 480, 3), dtype=np.uint8),
                    f"Camera {i+1} Error", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
            else:
                fpscam = cam.getFps()
                text = f'FPS: {fpscam:.2f}'
                (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                x, y = 15, 30
                cv2.rectangle(camFrame, (x - 8, y - textHeight - 8), (x + textWidth + 8, y + 8), (167, 80, 167), -1)
                cv2.putText(camFrame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, Color.White, 1)
                frames.append(camFrame)

        if len(frames):
            combined = tileImages(frames)
        else:
            combined = cv2.putText(
                np.zeros((360, 480, 3), dtype=np.uint8),
                "All Camera Error", (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        frame = resizeWithPadding(combined)
            
        originalImage = frame.copy()
        frameCount += 1
        currentTime = time.time()
        if currentTime - prevTime >= 1:
            fps = frameCount
            frameCount = 0
            prevTime = currentTime
            
        frame.flags.writeable = False
        bboxes, kpss = detModel.detect(frame, max_num=2)
        frame.flags.writeable = True
        rects = []

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4].astype(int)
            score = box[4]
            if score < 0.6:
                continue
            faceWidth = x2 - x1
            faceHeight = y2 - y1
            if faceWidth >= 30:
                rects.append((x1, y1, faceWidth, faceHeight, i))

        objects = tracker.update(rects)
        if len(objects):
            sleepTimer = datetime.now() + timedelta(minutes=1)
            if not worked:
                worked = True
            for objectId, (x, y, w, h, idx) in objects.items():
                if objectId not in faceJobStates:
                    faceJobStates[objectId] = manager.dict({
                        "step": 0,
                        "result": manager.list(),
                        "code": None,
                        "last_frame": None,
                        "success": False,
                        "list_face": manager.list(),
                        "pid": None,
                        "is_run": False
                    })
                if objectId in faceJobStates:
                    state = faceJobStates[objectId]
                    filteredIds = [objId for objId, data in faceJobStates.items() if data['success'] == False]
                    minObjectId = min(filteredIds) if filteredIds else None
                    if objectId == minObjectId and not startThread.is_alive():
                        faceCrop = originalImage[y:y+h, x:x+w]
                        light = checkLight(faceCrop)
                        if light:
                            if len(state["list_face"]) < countThread:
                                if 0 <= idx < len(bboxes):
                                    box = bboxes[idx]
                                    kps = kpss[idx]
                                    alignedFace = face_align.norm_crop(originalImage.copy(), landmark=kps)
                                    success, encoded = cv2.imencode(".png", alignedFace)
                                    if success:
                                        faceJobStates[objectId]["list_face"].append(encoded.tobytes())
                            elif not state["is_run"]:
                                faceJobStates[objectId]["is_run"] = True
                                startThread = Process(target=runRecognitionJob, args=(objectId, faceJobStates,))
                                startThread.daemon = True
                                startThread.start()
                            elif state["step"] >= countThread:
                                names = state["result"]
                                if len(names) == countThread and names.count(None) == 0 and len(set(names)) == 1:
                                    if not state["success"]:
                                        if objectId in faceJobStates:
                                            faceJobStates[objectId]["success"] = True

                                        imageBytes = np.frombuffer(faceJobStates[objectId]["list_face"][1], dtype=np.uint8)
                                        faceImage = cv2.imdecode(imageBytes, cv2.IMREAD_COLOR)
                                        updateInfo(state["code"], faceImage)
                                        log(f"[ID {objectId}] Thành công: {names[0]}")
                                else:
                                    log(f"[ID {objectId}] → reset lại")
                                    if objectId in faceJobStates:
                                        faceJobStates[objectId] = manager.dict({
                                            "step": 0,
                                            "result": manager.list(),
                                            "code": None,
                                            "last_frame": None,
                                            "success": False,
                                            "list_face": manager.list(),
                                            "pid": None,
                                            "is_run": False
                                        })
                            else:
                                log(f"[ID {objectId}] wait {datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]}")
                        else:
                            turnOn()
                # cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0), 2)
        else:
            if worked:
                worked = False
                tracker.reset()
                for objectId in faceJobStates.keys():
                    killJob(objectId)
                faceJobStates = manager.dict()
                minObjectId = None
                textSizeCache = {}

        frame = resizeWithPadding(frame)

        if sleepTimer >= datetime.now():
            text = f'FPS: {fps} - ID: {minObjectId}'
            (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            x, y = 15, 30
            cv2.rectangle(frame, (x - 8, y - textHeight - 8), (x + textWidth + 8, y + 8), (167, 80, 167), -1)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, Color.White, 1)
        else:
            turnOff()


        opencvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capturedImage = Image.fromarray(opencvImage)
        photoImage = ImageTk.PhotoImage(image=capturedImage)

        labelWidget.photoImage = photoImage
        labelWidget.configure(image=photoImage)
        labelWidget.after(1, handleImage, labelWidget, tracker)
    except Exception as e:
        log(f"Error handleImage: {e}", True)
        # textToSpeechSleep("Đã xảy ra lỗi xử lý hình ảnh", 1.2, False)
        handleImage(labelWidget, tracker)

def addNewEntry(code, name, image, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")

    global dataList
    name, lineBreak = name
    newEntry = {
        "image": image,
        "code": code,
        "name": name,
        "lineBreak": lineBreak,
        "timestamp": timestamp
    }
    with dataListLock:
        
        dataList = [entry for entry in dataList if entry["code"] != code]
        dataList.insert(0, newEntry)

        if len(dataList) > 6:
            dataList.pop()

        renderListUsers(canvas, dataList)

def renderListUsers(canvas, dataList, spacing=78):
    positions = []

    for i in range(6):
        yOffset = i * spacing
        positions.append({
            "imgY": 86 + yOffset,
            "imgTY": 109 + yOffset,
            "titleCodeY": 61 + yOffset + 5,
            "codeY": 60 + yOffset + 5,
            "nameY": 80 + yOffset + 5,
            "timeY": 102.5 + yOffset - 2
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
            pilImage = Image.open(entry["image"])
        except Exception as e:
            try:
                pilImage = Image.open("assets/images/facialRecognitionInterface/image_6.png")
            except Exception as e2:
                print(f"Error loading fallback image: {e2}")
                continue

        tkImgT = ImageTk.PhotoImage(file="assets/images/facialRecognitionInterface/image_3.png")
        pilImage = pilImage.resize((65, 65))
        tkImage = ImageTk.PhotoImage(pilImage)

        imgId = canvas.create_image(688, pos["imgY"], image=tkImage)
        imgTId = canvas.create_image(688, pos["imgTY"], image=tkImgT)

        titleCode = canvas.create_text(725, pos["titleCodeY"], anchor="nw", text="Mã NV:", fill="#767676", font=("Amiko", -14, "bold"))
        codeId = canvas.create_text(788, pos["codeY"], anchor="nw", text=entry["code"], fill="#000000", font=("Amiko", -15, "bold"))
        nameId = canvas.create_text(725, pos["nameY"], anchor="nw", text=entry["name"], fill="#000000", font=("Amiko", -16, "bold"))
        timeId = canvas.create_text(673.5, pos["timeY"], anchor="nw", text=entry["timestamp"], fill="#EBBB36", font=("Amiko", -13, "bold"))

        canvas._photoRefs.extend([tkImage, tkImgT])
        canvas._infoItems.extend([imgId, imgTId, titleCode, codeId, nameId, timeId])

def openCam():
    labelWidget = Label(window)
    labelWidget.place(x=15, y=66)
    tracker = CentroidTracker(maxDisappeared=5, maxDistance=60, useCache=True, cacheLifetime=1.5, killJob=killJob)
    try:
        if not checkInternet():
            for cam in cameras:
                cam.stop()
                cam.join()
            cv2.destroyAllWindows()
            time.sleep(0.5)
            textToSpeechSleep("./noInternet.wav", 1.2)
            return
        else:
            handleImage(labelWidget, tracker)

    except Exception as e:
        log(f"Error in videoCapture: {e}", True)
        turnOff()
        for cam in cameras:
            cam.stop()
            cam.join()
        cv2.destroyAllWindows()
        time.sleep(0.5)

if __name__ == "__main__":
    font = ImageFont.truetype('fonts/Montserrat-SemiBold.otf', 20)
    textSizeCache = {}
    window = Tk()
    window.title("Face Recognition by TMS")
    window.geometry("1024x535")
    window.configure(bg = "#FFFFFF")

    canvas = Canvas(window, bg="#FFFFFF", height=535, width=1024, bd=0, highlightthickness=0, relief="ridge")
    canvas.place(x=0, y=0)

    imageImage1 = PhotoImage(file="assets/images/facialRecognitionInterface/image_1.png")
    image1 = canvas.create_image(822.0, 267.0, image=imageImage1)

    imageImage4 = PhotoImage(file="assets/images/facialRecognitionInterface/image_4.png")
    image4 = canvas.create_image(317.0, 293.0, image=imageImage4)

    imageImage5 = PhotoImage(file="assets/images/facialRecognitionInterface/image_5.png")
    image5 = canvas.create_image(317.0, 34.0, image=imageImage5)

    canvas.create_text(751.0, 27.0, anchor="nw", text=datetime.now().strftime('%d-%m-%Y'), fill="#000000", font=("Amiko", -15, "bold"))

    cameras = []

    for camId in [0, 2]:
    # for camId in getIndexCam():
        cam = CameraReader(camId)
        cam.start()
        cameras.append(cam)

    openCam()

    window.resizable(True, True)

    def onClosing():
        turnOff()
        for cam in cameras:
            cam.stop()
            cam.join()
        cv2.destroyAllWindows()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", onClosing)
    window.mainloop()