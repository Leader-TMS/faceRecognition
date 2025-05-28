import cv2
import random
import time
import threading
import mediapipe as mp
from collections import OrderedDict
from datetime import datetime
import numpy as np
import os
import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
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
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play, _play_with_ffplay
import io
import string
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import subprocess
import signal
from multiprocessing import Process, Manager
import socket
from facesController import CentroidTracker
import requests
from PIL import ImageFont, ImageDraw, Image, ImageTk
from tkinter import Tk, Canvas, PhotoImage, Label
import faiss
from insightface.model_zoo import get_model
from insightface.utils import face_align

tracedModel = torch.jit.load('tracedModel/inceptionResnetV1Traced.pt')
torch.set_grad_enabled(False)
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')
index = faiss.read_index('index/faiss.index')
ids = joblib.load('index/ids.pkl')

recModel = get_model('models/w600k_mbf.onnx')
recModel.prepare(ctx_id=-1, input_size=(640, 640))

faissIndex = faiss.read_index('indexWithInsight/faiss.index')
labelList = joblib.load('indexWithInsight/labels.pkl')

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
urlGetEmbedding = config['settingApi']["URL_GET_EMBEDDING"]

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
frameCount = 0
prevTime = time.time()
fps = 0
sleepTimer = datetime.now()
minObjectId = None
worked = False
session = requests.Session() 
dataListGlobal = []
NamedTemporaryFile
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
        
        for index in devVideo:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return index
            else:
                cap.release()
        return None
    except Exception as e:
        log(f"getIndexCam: {e}", True)
        return None

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
        if objectId in trackName: 
            del trackName[objectId]
        if objectId in faceSpeedTrackers: 
            del faceSpeedTrackers[objectId]
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

def getEmbedding(imageVar, serverUrl):
    try:
        global session
        if not isinstance(imageVar, np.ndarray):
            return {'error': 'imageVar phải là numpy.ndarray (OpenCV RGB image)'}

        _, imgEncoded = cv2.imencode('.jpg', imageVar, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        imgBytes = io.BytesIO(imgEncoded.tobytes())
        imgBytes.seek(0)
        files = {'image': ('image.jpg', imgBytes, 'image/jpeg')}

        if session is None:
            log('session: none')
            session = requests.Session()

        response = session.post(serverUrl, files=files, timeout=3)

        if response.ok:
            result = response.json()
            if 'embedding' in result:
                return result
            else:
                return {'error': 'Không có embedding trong response từ server'}
        else:
            return {'error': 'Request failed', 'status': response.status_code}
            
    except Exception as e:
        log(e, True)
        return {'error': str(e)}

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
                    employeeCode = labelList[I[0][0]]
                    employeeCode = getEmployeesByCode(employeeCode)
                    if employeeCode:
                        label = employeeCode['short_name']
                        employeeCode = employeeCode['employee_code']
                        saveFaceDirection(faceImage, f"evidences/almostSimilar/{employeeCode}_{label}")
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

def displayText(frame, x, y, w, light, smallFace, angle, lowSpeed, blur, color):
    try:
        return frame
        x = x + w
        conditions = [
            (not light, "Thiếu sáng", (0, 0, 0)),
            (smallFace, "Khuôn mặt nhỏ", (71, 173, 253)),
            (not angle, "Nhìn thẳng", (0, 0, 255)),
            (not lowSpeed, "Chậm lại", (255, 0, 0)),
            (blur, "Ảnh bị mờ", (167, 80, 167))
        ]

        textY = y - 10

        for condition, text, rectColor in conditions:
            if condition:
                try:
                    (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    frame = drawTextVietnamese(frame, text, (x, textY), background=rectColor)
                    textY += textHeight + 18
                except Exception as e:
                    log(f"Error drawing text '{text}' at position ({x}, {textY}): {e}", True)
                    continue

        return frame
    
    except Exception as e:
        log(f"Error in displayText: {e}", True)
        return frame

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

def checkBlurry(image, threshold=100):
    try:
        if image is not None and image.size > 0:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(grayImage, cv2.CV_64F)
            variance = laplacian.var()
            if variance < threshold:
                return True
        return False
    except cv2.error as e:
        log(f"OpenCV error in checkBlurry: {e}", True)
        return True
    except Exception as e:
        log(f"Unexpected error in checkBlurry: {e}", True)
        return True

def faceAngle(nose, leftEye, rightEye, fWidth, fHeight):
    try:
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

    except Exception as e:
        log(f"Error in faceAngle calculation: {e}", True)
        return False

def goodFaceAngle(image):
    goodAngle = False
    try:
        if image is not None and image.size > 0:
            height, width = image.shape[:2]
            results = faceMesh.process(image)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    leftEye = landmarks.landmark[173]
                    rightEye = landmarks.landmark[398]
                    nose = landmarks.landmark[4]
                    goodAngle = faceAngle(nose, leftEye, rightEye, width, height)
        return goodAngle
    except Exception as e:
        log(f"Error in goodFaceAngle: {e}", True)
        return False

# def motionBlurCompensation(image):
#     try:
#         blurred = cv2.GaussianBlur(image, (21, 21), 0)
#         sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
#         return sharpened
#     except cv2.error as e:
#         log(f"OpenCV error in motionBlurCompensation: {e}", True)
#         return None
#     except Exception as e:
#         log(f"Unexpected error in motionBlurCompensation: {e}", True)
#         return None

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

            pathImage = saveFaceDirection(image, f"evidences/valid/{shortName}")

            thread = threading.Thread(
                target=addNewEntry,
                kwargs={
                    'code': employeeCode,
                    'name': wrapText(fullName),
                    'image': pathImage if pathImage else "assets/images/facialRecognitionInterface/image_6.png"
                }
            )
            thread.daemon = True
            thread.start()
            # addNewEntry(code=employeeCode, name=wrapText(fullName), image=pathImage if pathImage else "assets/images/facialRecognitionInterface/image_6.png")

            saveAttendance("face", employeeCode, genUniqueId())
                # textToSpeech("Lỗi lưu dữ liệu server", 1.2)
            
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

# def waitFullName():
#     global resetCallTTS, objCheckName, stopThread
#     try:
#         if stopThread:
#             stopThread = False
#         while not stopThread:
#             if checkTime(resetCallTTS, 0.5):
#                 if len(objCheckName):
#                     listName = checkFormatTXT(objCheckName)
#                     if listName != '':
#                         log(f"Xin chào {listName} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
#                         textToSpeech(f"Xin chào {listName}", 1.2, personName)
#                 objCheckName = {}
#                 resetCallTTS = datetime.now()
#                 stopThread = False
#                 break
#             time.sleep(0.1)
#     except Exception as e:
#         log(f"Error in waitFullName: {e}", True)
#         stopThread = False

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

def checkFormatTXT(objCheckName):
    try:
        if len(objCheckName) >= 2:
            value = ', '.join(objCheckName.values())
        else:
            value = next(iter(objCheckName.values()), '')
        return value
    except Exception as e:
        log(f"Error in checkFormatTXT: {e}", True)
        return ''

def checkTime(savedTime, second=0.3):
    try:
        return round(((datetime.now() - savedTime).total_seconds()), 2) >= second
    except Exception as e:
        log(f"Error in checkTime: {e}", True)
        return False

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
        # play(audio)

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

def corruptImageDetected(image):
    try:
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
    except Exception as e:
        log(f"Error in corruptImageDetected: {e}", True)
        return True

def drawTextVietnamese(img, text, position, color=(255, 255, 255), background=(65, 169, 35), cache=False):
    try:
        return img
        imgPIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(imgPIL)
        if cache and text in textSizeCache:
            textWidth, textHeight = textSizeCache[text]
        else:
            bbox = draw.textbbox((0, 0), text, font=font)
            textWidth = bbox[2] - bbox[0]
            textHeight = bbox[3] - bbox[1]
            if cache:
                textSizeCache[text] = (textWidth, textHeight)

        x, y = position

        draw.rectangle([(x - 15, y - textHeight - 12), (x + textWidth + 12, y)], fill=background)

        draw.text((x, y - textHeight - 10), text, font=font, fill=color)

        # draw.text(position, text, font=font, fill=color)

        return cv2.cvtColor(np.array(imgPIL), cv2.COLOR_RGB2BGR)
    except Exception as e:
        log(f"Error in drawTextVietnamese: {e}", True)
        return img

def videoCapture():
    global faceJobStates, trackName, faceSpeedTrackers, startThread
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE , -4)
    tracker = CentroidTracker(maxDisappeared=5, maxDistance=60, useCache=True, cacheLifetime=1.5, killJob=killJob)
    frameCount = 0
    prevTime = time.time()
    fps = 0
    sleepTimer = datetime.now()
    minObjectId = None
    worked = False
    try:
        if not checkInternet():
            cap.release()
            cv2.destroyAllWindows()
            textToSpeechSleep("./noInternet.wav", 1.2)

        if not cap.isOpened():
            cap.release()
            cv2.destroyAllWindows()
            textToSpeechSleep("Không tìm thấy máy ảnh, Vui lòng kiểm tra lại", 1.2)
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                textToSpeechSleep("Máy ảnh bị mất tín hiệu, Bắt đầu khởi động lại", 1.2)
                break
            else:
                
                frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)

                originalImage = frame.copy()
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
                    for detection in results.detections:
                        score = detection.score[0]
                        if score >= 0.75:
                            bboxC = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            x = int(bboxC.xmin * w)
                            y = int(bboxC.ymin * h)
                            width = int(bboxC.width * w)
                            height = int(bboxC.height * h)
                            if width >= 55:
                                rects.append((x, y, width, height))
                                if  width >= 55 and width < 65:
                                        frame = displayText(frame, x, y, width, True, True, True, True, False, Color)

                objects = tracker.update(rects)
                if len(objects):
                    sleepTimer = datetime.now() + timedelta(minutes=1)
                    if not worked:
                        worked = True
                    for objectId, (x, y, w, h) in objects.items():
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
                                _, faceWidth = faceCrop.shape[:2]
                                light = checkLight(faceCrop)
                                if light:
                                    if faceWidth >= 65:
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
                                        angle = goodFaceAngle(faceCrop.copy())
                                        lowSpeed = normSpeed < 0.2

                                        if len(state["list_face"]) < countThread:
                                            if angle and lowSpeed:
                                                if blur:
                                                    frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)
                                                else:
                                                    corruptImage = corruptImageDetected(faceCrop)
                                                    if corruptImage:
                                                        frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, corruptImage, Color)
                                                    else:
                                                        success, encoded = cv2.imencode(".png", faceCrop.copy())
                                                        if success:
                                                            faceJobStates[objectId]["list_face"].append(encoded.tobytes())
                                                        # faceJobStates[objectId]["list_face"].append(faceCrop.copy())
                                            else:
                                                frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)
                                        elif not state["is_run"]:
                                            faceJobStates[objectId]["is_run"] = True
                                            startThread = Process(target=runRecognitionJob, args=(objectId, faceJobStates,))
                                            startThread.daemon = True
                                            startThread.start()
                                        elif state["step"] >= countThread:
                                            names = state["result"]
                                            if len(names) == countThread and names.count(None) == 0 and len(set(names)) == 1:
                                                trackName[objectId] = names[0]
                                                if not state["success"]:
                                                    if objectId in faceJobStates:
                                                        faceJobStates[objectId]["success"] = True
                                                    updateInfo(state["code"], faceCrop)
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
                                    turnOn()

                            name = trackName.get(objectId, "Đang kiểm tra..." if state["step"] > 0 else "Người mới")
                            if(state["success"]):
                                frame = drawTextVietnamese(frame, name, (x, y))
                            else:
                                if name == "Người mới":
                                    bColor = (255, 0, 0)
                                elif name == "Đang kiểm tra...":
                                    bColor = (249, 172, 75)
                                else:
                                    bColor = (65, 169, 35)
                                frame = displayText(frame, x, y, w, light, False, True, True, False, Color)

                                frame = drawTextVietnamese(frame, name, (x, y), background=bColor)
                else:
                    if worked:
                        worked = False
                        tracker.reset()
                        for objectId in faceJobStates.keys():
                            killJob(objectId)
                        faceJobStates = manager.dict()
                        trackName = {}
                        faceSpeedTrackers = {}
                        minObjectId = None

                if sleepTimer >= datetime.now():
                    #subprocess.run(["xset", "dpms", "force", "on"])
                    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Id check: {minObjectId}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
                    turnOff()
                    break
        turnOff()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        log(f"Error in videoCapture: {e}", True)
        turnOff()
        cap.release()
        cv2.destroyAllWindows()

def getFaceLandmarks(frame, resultsDetection, faceMesh, displayText, Color):
    hFrame, wFrame, _ = frame.shape
    kpss, bboxes, rects = [], [], []
    t0 = time.time()
    if resultsDetection.detections:
        for i, detection in enumerate(resultsDetection.detections):
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * wFrame)
            y = int(bboxC.ymin * hFrame)
            w = int(bboxC.width * wFrame)
            h = int(bboxC.height * hFrame)

            x1Crop = max(0, x)
            y1Crop = max(0, y)
            x2Crop = min(wFrame, x + w)
            y2Crop = min(hFrame, y + h)

            faceCrop = frame[y1Crop:y2Crop, x1Crop:x2Crop]
            faceRgb = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2RGB)
            resultsMeshCrop = faceMesh.process(faceRgb)

            if resultsMeshCrop.multi_face_landmarks:
                for faceLandmarks in resultsMeshCrop.multi_face_landmarks:
                    lm = faceLandmarks.landmark
                    kps = np.array([
                        [lm[468].x * (x2Crop - x1Crop) + x1Crop, lm[468].y * (y2Crop - y1Crop) + y1Crop],
                        [lm[473].x * (x2Crop - x1Crop) + x1Crop, lm[473].y * (y2Crop - y1Crop) + y1Crop],
                        [lm[4].x * (x2Crop - x1Crop) + x1Crop, lm[4].y * (y2Crop - y1Crop) + y1Crop],
                        [lm[61].x * (x2Crop - x1Crop) + x1Crop, lm[61].y * (y2Crop - y1Crop) + y1Crop],
                        [lm[291].x * (x2Crop - x1Crop) + x1Crop, lm[291].y * (y2Crop - y1Crop) + y1Crop],
                    ], dtype=np.float32)
                    kpss.append(kps)

                    xCoords, yCoords = kps[:, 0], kps[:, 1]
                    x1, y1 = np.min(xCoords), np.min(yCoords)
                    x2, y2 = np.max(xCoords), np.max(yCoords)

                    padding = 1.3
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    wBox, hBox = (x2 - x1) * (1 + padding), (y2 - y1) * (1 + padding)
                    x1New = int(max(0, cx - wBox / 2))
                    y1New = int(max(0, cy - hBox / 2))
                    x2New = int(min(wFrame - 1, cx + wBox / 2))
                    y2New = int(min(hFrame - 1, cy + hBox / 2))

                    bbox = np.array([x1New, y1New, x2New, y2New, 1.0], dtype=np.float32)
                    bboxes.append(bbox)
                    rects.append((x1New, y1New, x2New - x1New, y2New - y1New, i))
        print(f'[resultsDetection] {time.time() - t0}')
    return frame, kpss, bboxes, rects

def getFaceLandmarks(frame, resultsDetection, faceMesh, displayText, Color):
    hFrame, wFrame, _ = frame.shape
    kpss, bboxes, rects = [], [], []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultsMeshFull = faceMesh.process(frame_rgb)
    if resultsDetection.detections:
        for i, detection in enumerate(resultsDetection.detections):
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * wFrame)
            y = int(bboxC.ymin * hFrame)
            w = int(bboxC.width * wFrame)
            h = int(bboxC.height * hFrame)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(wFrame, x + w), min(hFrame, y + h)

            matched = False
            if resultsMeshFull.multi_face_landmarks:
                for lm in resultsMeshFull.multi_face_landmarks:
                    cx = int(np.mean([p.x for p in lm]) * wFrame)
                    cy = int(np.mean([p.y for p in lm]) * hFrame)
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        matched = True
                        keypoints = np.array([
                            [lm[468].x * wFrame, lm[468].y * hFrame],
                            [lm[473].x * wFrame, lm[473].y * hFrame],
                            [lm[4].x * wFrame, lm[4].y * hFrame],
                            [lm[61].x * wFrame, lm[61].y * hFrame],
                            [lm[291].x * wFrame, lm[291].y * hFrame],
                        ], dtype=np.float32)
                        break

            if not matched:
                faceCrop = frame[y1:y2, x1:x2]
                if faceCrop.size == 0:
                    continue
                faceRgb = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2RGB)
                resultsMeshCrop = faceMesh.process(faceRgb)
                if resultsMeshCrop.multi_face_landmarks:
                    lm = resultsMeshCrop.multi_face_landmarks[0].landmark
                    keypoints = np.array([
                        [lm[468].x * (x2 - x1) + x1, lm[468].y * (y2 - y1) + y1],
                        [lm[473].x * (x2 - x1) + x1, lm[473].y * (y2 - y1) + y1],
                        [lm[4].x * (x2 - x1) + x1, lm[4].y * (y2 - y1) + y1],
                        [lm[61].x * (x2 - x1) + x1, lm[61].y * (y2 - y1) + y1],
                        [lm[291].x * (x2 - x1) + x1, lm[291].y * (y2 - y1) + y1],
                    ], dtype=np.float32)
                else:
                    continue

            xCoords, yCoords = keypoints[:, 0], keypoints[:, 1]
            x1kp, y1kp = np.min(xCoords), np.min(yCoords)
            x2kp, y2kp = np.max(xCoords), np.max(yCoords)

            padding = 1.3
            cx, cy = (x1kp + x2kp) / 2, (y1kp + y2kp) / 2
            wBox, hBox = (x2kp - x1kp) * (1 + padding), (y2kp - y1kp) * (1 + padding)
            x1New = int(max(0, cx - wBox / 2))
            y1New = int(max(0, cy - hBox / 2))
            x2New = int(min(wFrame - 1, cx + wBox / 2))
            y2New = int(min(hFrame - 1, cy + hBox / 2))

            faceWidth = x2New - x1New
            bbox = np.array([x1New, y1New, x2New, y2New, 1.0], dtype=np.float32)
            bboxes.append(bbox)
            kpss.append(keypoints)

            if faceWidth >= 20:
                rects.append((x1New, y1New, faceWidth, y2New - y1New, i))
                if 20 <= faceWidth < 65:
                    frame = displayText(frame, x1New, y1New, faceWidth, True, True, True, True, False, Color)

    return frame, kpss, bboxes, rects

def getFaceLandmarks(frame, resultsMesh, displayText, Color):
    hFrame, wFrame, _ = frame.shape
    kpss, bboxes, rects = [], [], []


    if resultsMesh.multi_face_landmarks:
        for i, lm in enumerate(resultsMesh.multi_face_landmarks):
            try:
                keypoints = np.array([
                    [lm.landmark[468].x * wFrame, lm.landmark[468].y * hFrame],
                    [lm.landmark[473].x * wFrame, lm.landmark[473].y * hFrame],
                    [lm.landmark[4].x * wFrame, lm.landmark[4].y * hFrame],
                    [lm.landmark[61].x * wFrame, lm.landmark[61].y * hFrame],
                    [lm.landmark[291].x * wFrame, lm.landmark[291].y * hFrame],
                ], dtype=np.float32)
            except IndexError:
                continue

            kpss.append(keypoints)

            xCoords, yCoords = keypoints[:, 0], keypoints[:, 1]
            x1, y1 = np.min(xCoords), np.min(yCoords)
            x2, y2 = np.max(xCoords), np.max(yCoords)

            padding = 1.3
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wBox, hBox = (x2 - x1) * (1 + padding), (y2 - y1) * (1 + padding)
            x1New = int(max(0, cx - wBox / 2))
            y1New = int(max(0, cy - hBox / 2))
            x2New = int(min(wFrame - 1, cx + wBox / 2))
            y2New = int(min(hFrame - 1, cy + hBox / 2))

            faceWidth = x2New - x1New
            bbox = np.array([x1New, y1New, x2New, y2New, 1.0], dtype=np.float32)
            bboxes.append(bbox)

            if faceWidth >= 20:
                rects.append((x1New, y1New, faceWidth, y2New - y1New, i))
                if 20 <= faceWidth < 65:
                    frame = displayText(frame, x1New, y1New, faceWidth, True, True, True, True, False, Color)

    return frame, kpss, bboxes, rects

def handleImage(camera, labelWidget, faceDetection, faceMesh, tracker):
    global faceJobStates, trackName, faceSpeedTrackers, startThread, frameCount, prevTime, fps, sleepTimer, minObjectId, worked, textSizeCache
    try:
        openCamera = time.time()
        ret, frame = camera.read()
        
        if not ret:
            camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)
            textToSpeechSleep("Máy ảnh bị mất tín hiệu, Bắt đầu khởi động lại", 1.2)
            return
        elif frame is not None and frame.size:
            t0 = time.time()
            print(f"[openCamera] {t0 - openCamera:.4f} sec")
            # frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)

            originalImage = frame.copy()
            frameCount += 1
            currentTime = time.time()
            if currentTime - prevTime >= 1:
                fps = frameCount
                frameCount = 0
                prevTime = currentTime

            # h, w = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t1 = time.time()
            print(f"[Resize + BGR->RGB] {t1 - t0:.4f} sec")
            frame.flags.writeable = False
            # resultsDetection = faceDetection.process(rgbFrame)
            resultsMesh = faceMesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t2 = time.time()
            print(f"[FaceMesh] {t2 - t1:.4f} sec")
            frame, kpss, bboxes, rects = getFaceLandmarks(
                frame,
                resultsMesh=resultsMesh,
                # resultsDetection=resultsDetection, 
                # faceMesh=faceMesh,
                displayText=displayText,
                Color=Color
            )

            # bboxes = []
            # kpss = []
            # rects = []
            # hFrame, wFrame = frame.shape[:2]
            # if resultsDetection.detections:
            #     for i, detection in enumerate(resultsDetection.detections):
            #         bboxC = detection.location_data.relative_bounding_box
            #         ih, iw, _ = frame.shape
            #         x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            #         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # if results.multi_face_landmarks:
            #     for i, faceLandmarks in enumerate(results.multi_face_landmarks):
            #         lm = faceLandmarks.landmark
            #         kps = np.array([
            #             [lm[468].x * wFrame,  lm[468].y * hFrame],    # 0 - Left eye
            #             [lm[473].x * wFrame, lm[473].y * hFrame],   # 1 - Right eye
            #             [lm[4].x * wFrame,  lm[4].y * hFrame],    # 2 - Nose
            #             [lm[61].x * wFrame,  lm[61].y * hFrame],    # 3 - Left mouth
            #             [lm[291].x * wFrame, lm[291].y * hFrame],   # 4 - Right mouth
            #         ], dtype=np.float32)
            #         kpss.append(kps)
            #         x_coords = kps[:, 0]
            #         y_coords = kps[:, 1]
            #         x1, y1 = np.min(x_coords), np.min(y_coords)
            #         x2, y2 = np.max(x_coords), np.max(y_coords)

            #         padding = 1.3
            #         cx = (x1 + x2) / 2
            #         cy = (y1 + y2) / 2
            #         w_box = (x2 - x1) * (1 + padding)
            #         h_box = (y2 - y1) * (1 + padding)
            #         x1_new = int(max(0, cx - w_box / 2))
            #         y1_new = int(max(0, cy - h_box / 2))
            #         x2_new = int(min(wFrame - 1, cx + w_box / 2))
            #         y2_new = int(min(hFrame - 1, cy + h_box / 2))

            #         bbox = np.array([x1_new, y1_new, x2_new, y2_new, 1.0], dtype=np.float32)
            #         bboxes.append(bbox)
            #         faceWidth = x2_new - x1_new
            #         print(f'faceWidth: {faceWidth}')
            #         if faceWidth >= 20:
            #             rects.append((x1_new, y1_new, x2_new - x1_new, y2_new - y1_new, i))
            #             if  faceWidth >= 20 and faceWidth < 65:
            #                 frame = displayText(frame, x1_new, y1_new, faceWidth, True, True, True, True, False, Color)

            bboxes = np.array(bboxes)
            kpss = np.array(kpss)
            t3 = time.time()
            print(f"[Bboxes + Kpss] {t3 - t2:.4f} sec")
            t4 = time.time()
            print(f"[alignedFace + embedding] {t4 - t3:.4f} sec")
            print(f"[total] {t4 - t0:.4f} sec \n")
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
                            # _, faceWidth = faceCrop.shape[:2]
                            light = checkLight(faceCrop)
                            if light:
                                # if faceWidth >= 20:
                                frameHeight, frameWidth = frame.shape[:2]
                                centerXPixel = x + w // 2
                                centerYPixel = y + h // 2
                                centerXNorm = centerXPixel / frameWidth
                                centerYNorm = centerYPixel / frameHeight

                                currentSpeedTime = time.time()
                                centerNorm = (centerXNorm, centerYNorm)
                                centerPixel = (centerXPixel, centerYPixel)

                                normSpeed = 0
                                # pixelSpeed = 0

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

                                    faceSpeedTrackers[objectId] = {
                                        "prev_norm": centerNorm,
                                        "prev_pixel": centerPixel,
                                        "time": currentSpeedTime
                                    }
                                
                                blur = checkBlurry(faceCrop)
                                angle = goodFaceAngle(faceCrop.copy())
                                lowSpeed = normSpeed < 0.2
                                if len(state["list_face"]) < countThread:
                                    if angle and lowSpeed:
                                        if blur:
                                            frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)
                                        else:
                                            corruptImage = corruptImageDetected(faceCrop)
                                            if corruptImage:
                                                frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, corruptImage, Color)
                                            else:
                                                if 0 <= idx < len(bboxes):
                                                    box = bboxes[idx]
                                                    # x1, y1, x2, y2 = box[:4].astype(int)
                                                    kps = kpss[idx]
                                                    alignedFace = face_align.norm_crop(originalImage.copy(), landmark=kps)
                                                    success, encoded = cv2.imencode(".png", alignedFace)
                                                    if success:
                                                        faceJobStates[objectId]["list_face"].append(encoded.tobytes())
                                    else:
                                        frame = displayText(frame, x, y, w, light, False, angle, lowSpeed, blur, Color)
                                elif not state["is_run"]:
                                    faceJobStates[objectId]["is_run"] = True
                                    startThread = Process(target=runRecognitionJob, args=(objectId, faceJobStates,))
                                    startThread.daemon = True
                                    startThread.start()
                                elif state["step"] >= countThread:
                                    names = state["result"]
                                    if len(names) == countThread and names.count(None) == 0 and len(set(names)) == 1:
                                        trackName[objectId] = names[0]
                                        if not state["success"]:
                                            if objectId in faceJobStates:
                                                faceJobStates[objectId]["success"] = True
                                            updateInfo(state["code"], faceCrop)
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

                        name = trackName.get(objectId, "Đang kiểm tra..." if state["step"] > 0 else "Người mới")

                        if(state["success"]):
                            cv2.rectangle(frame, (x, y), (x+w,y+h), Color.Green, 2)
                        else:
                            if name == "Người mới":
                                lColor = (0, 0, 255)
                            elif name == "Đang kiểm tra...":
                                lColor = (75, 172, 249)
                            else:
                                lColor = (65, 169, 35)
                            cv2.rectangle(frame, (x, y), (x+w,y+h), lColor, 2)

                        cv2.putText(frame, f'{objectId}', (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, Color.Green, 1)
            else:
                if worked:
                    worked = False
                    tracker.reset()
                    for objectId in faceJobStates.keys():
                        killJob(objectId)
                    faceJobStates = manager.dict()
                    trackName = {}
                    faceSpeedTrackers = {}
                    minObjectId = None
                    textSizeCache = {}

            if sleepTimer >= datetime.now():
                text = f'FPS: {fps} - ID: {minObjectId}'
                (textWidth, textHeight), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                x, y = 15, 30
                cv2.rectangle(frame, (x - 8, y - textHeight - 8), (x + textWidth + 8, y + 8), (167, 80, 167), -1)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, Color.White, 1)
            else:
                turnOff()

            frame = cv2.resize(frame, (605, 454), interpolation=cv2.INTER_LANCZOS4)

        opencvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        capturedImage = Image.fromarray(opencvImage)
        photoImage = ImageTk.PhotoImage(image=capturedImage)

        labelWidget.photoImage = photoImage
        labelWidget.configure(image=photoImage)
        labelWidget.after(1, handleImage, camera, labelWidget, faceDetection, faceMesh, tracker)
    except Exception as e:
        print(e)
        log(f"Error handleImage: {e}", True)
        # textToSpeechSleep("Đã xảy ra lỗi xử lý hình ảnh", 1.2, False)
        handleImage(camera, labelWidget, faceDetection, faceMesh, tracker)

def addNewEntry(code, name, image, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")

    global dataListGlobal
    print(f'dataListGlobal: {dataListGlobal}')
    dataListGlobal = [entry for entry in dataListGlobal if entry["code"] != code]
    name, lineBreak = name
    newEntry = {
        "image": image,
        "code": code,
        "name": name,
        "lineBreak": lineBreak,
        "timestamp": timestamp
    }

    dataListGlobal.insert(0, newEntry)

    if len(dataListGlobal) > 6:
        dataListGlobal.pop()

    renderListUsers(canvas, dataListGlobal)

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
            print(f"Error loading image: {e}")
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
    indexCam = getIndexCam()
    if isinstance(indexCam, int):
        camera = cv2.VideoCapture(indexCam)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        camera.set(cv2.CAP_PROP_EXPOSURE , -4)
        mpFaceDetection = mp.solutions.face_detection
        mpFaceMesh = mp.solutions.face_mesh

        faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.6, model_selection=1)
        faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=3, refine_landmarks=True, 
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        tracker = CentroidTracker(maxDisappeared=5, maxDistance=60, useCache=True, cacheLifetime=1.5, killJob=killJob)

        try:
            if not checkInternet():
                camera.release()
                cv2.destroyAllWindows()
                time.sleep(0.5)
                textToSpeechSleep("./noInternet.wav", 1.2)
                return
            elif not camera.isOpened():
                camera.release()
                cv2.destroyAllWindows()
                time.sleep(0.5)
                textToSpeechSleep("Không tìm thấy máy ảnh, Vui lòng kiểm tra lại", 1.2)
                return
            else:
                handleImage(camera, labelWidget, faceDetection, faceMesh, tracker)

        except Exception as e:
            log(f"Error in videoCapture: {e}", True)
            turnOff()
            camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)
    else:
        textToSpeechSleep("Không tìm thấy máy ảnh, Vui lòng kiểm tra lại", 1.2)
        return

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

    openCam()


    window.resizable(True, True)

    def onClosing():
        turnOff()
        cv2.destroyAllWindows()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", onClosing)
    window.mainloop()