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
import string
from skimage.metrics import structural_similarity as ssim
from tempfile import NamedTemporaryFile
import subprocess
import signal
from multiprocessing import Process, Manager
import socket
from facesController import CentroidTracker
from collections import deque
import requests
from PIL import ImageFont, ImageDraw, Image, ImageTk
from pathlib import Path
from tkinter import Tk, Canvas, PhotoImage, Label
import faiss
from insightface.app import FaceAnalysis
tracedModel = torch.jit.load('tracedModel/inceptionResnetV1Traced.pt')
torch.set_grad_enabled(False)
svmModel = joblib.load('svmModel.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')
index = faiss.read_index('index/faiss.index')
ids = joblib.load('index/ids.pkl')

app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=-1)
index = faiss.read_index('index1/faiss.index')
ids = joblib.load('index1/labels.pkl')
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
            wFace = max(int(wFace * 0.4), 45)
            hFace = max(int(hFace * 0.4), 45)

            faceImage = cv2.resize(faceImage, (wFace, hFace), interpolation=cv2.INTER_LANCZOS4)

            faceRgb = cv2.cvtColor(faceImage, cv2.COLOR_BGR2RGB)

            start = time.time()

            embRes = getEmbedding(faceRgb, urlGetEmbedding)

            end = time.time()
            totalTime = end - start
            log(f'time get Embedding: {totalTime}')
            if 'error' in embRes:
                log(f"Embedding error: {embRes['error']}", True)
                mtcnn = MTCNN(margin=40, select_largest=False, selection_method='probability', keep_all=True, min_face_size=40, thresholds=[0.7, 0.8, 0.8])
                faceMtcnn = mtcnn(faceRgb)
                if faceMtcnn is not None and len(faceMtcnn) > 0:
                    for _, face in enumerate(faceMtcnn):
                        with torch.no_grad():
                            embedding = tracedModel(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            else:
                embedding = embRes["embedding"]

            if embedding is not None:
                start = time.time()
                # embedding = np.array(embedding, dtype='float32')
                # embedding = normalize(embedding.reshape(1, -1)).astype('float32')
                embedding = np.array(embedding, dtype='float32').reshape(1, -1)
                faiss.normalize_L2(embedding)
                D, I = index.search(embedding, 1)
                sim = D[0][0]
                log(f"sim: {sim}")
                log(f"search: {time.time() - start}")
                if sim >= 0.075:
                    employeeCode = ids[I[0][0]]
                    employeeCode = getEmployeesByCode(employeeCode)
                    if employeeCode:
                        label = employeeCode['short_name']
                        employeeCode = employeeCode['employee_code']
                elif sim >= 0.065:
                    employeeCode = ids[I[0][0]]
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
            # ---------------------------------
                # embedding = normalize([embedding])

                # labelIndex = svmModel.predict(embedding)[0]

                # prob = svmModel.predict_proba(embedding)[0]

                # probPercent = round(prob[labelIndex], 2)
                # print(f"probPercent: {probPercent}")
                # if probPercent >= 0.75:
                #     employeeCode = labelEncoder.inverse_transform([labelIndex])[0]
                #     employeeCode = getEmployeesByCode(employeeCode)
                #     if employeeCode:
                #         label = employeeCode['short_name']
                #         employeeCode = employeeCode['employee_code']
                # elif probPercent >= 0.65:
                #     employeeCode = labelEncoder.inverse_transform([labelIndex])[0]
                #     employeeCode = getEmployeesByCode(employeeCode)
                #     if employeeCode:
                #         label = employeeCode['short_name']
                #         employeeCode = employeeCode['employee_code']
                #         saveFaceDirection(faceImage, f"evidences/almostSimilar/{employeeCode}_{label}")
                #     label = None
                #     employeeCode = None
                # else:
                #     saveFaceDirection(faceImage, "evidences/invalid")
                #     label = None
                #     employeeCode = None
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
            # thread = threading.Thread(target=generateAndPlayAudio, args=(text, speed, typeId))
            # thread.daemon = True
            # thread.start()
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

def waitFullName():
    global resetCallTTS, objCheckName, stopThread
    try:
        if stopThread:
            stopThread = False
        while not stopThread:
            if checkTime(resetCallTTS, 0.5):
                if len(objCheckName):
                    listName = checkFormatTXT(objCheckName)
                    if listName != '':
                        log(f"Xin chào {listName} - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
                        textToSpeech(f"Xin chào {listName}", 1.2, personName)
                objCheckName = {}
                resetCallTTS = datetime.now()
                stopThread = False
                break
            time.sleep(0.1)
    except Exception as e:
        log(f"Error in waitFullName: {e}", True)
        stopThread = False

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

def handleImage(camera, labelWidget, faceDetection, tracker):
    global faceJobStates, trackName, faceSpeedTrackers, startThread, frameCount, prevTime, fps, sleepTimer, minObjectId, worked, textSizeCache
    try:
        ret, frame = camera.read()
        
        if not ret:
            camera.release()
            cv2.destroyAllWindows()
            time.sleep(0.5)
            textToSpeechSleep("Máy ảnh bị mất tín hiệu, Bắt đầu khởi động lại", 1.2)
            return
        elif frame is not None and frame.size:
            
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

                                            # pixelDist = math.hypot(
                                            #     centerPixel[0] - prevData["prev_pixel"][0],
                                            #     centerPixel[1] - prevData["prev_pixel"][1]
                                            # )
                                            # pixelSpeed = round(pixelDist / deltaTime, 2)

                                            # text = f"N:{normSpeed}/s | P:{pixelSpeed}px/s"
                                            # cv2.putText(frame, text, (x, y + h + 20),
                                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, Color.Orange, 2)

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
                            # frame = drawTextVietnamese(frame, name, (x, y), cache=True)
                            cv2.rectangle(frame, (x, y), (x+w,y+h), Color.Green, 2)
                        else:
                            if name == "Người mới":
                                lColor = (0, 0, 255)
                            elif name == "Đang kiểm tra...":
                                lColor = (75, 172, 249)
                            else:
                                lColor = (65, 169, 35)
                            cv2.rectangle(frame, (x, y), (x+w,y+h), lColor, 2)
                            # frame = displayText(frame, x, y, w, light, False, True, True, False, Color)
                            # frame = drawTextVietnamese(frame, name, (x, y), background=bColor, cache=True)
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
        labelWidget.after(1, handleImage, camera, labelWidget, faceDetection, tracker)
    except Exception as e:
        print(e)
        log(f"Error handleImage: {e}", True)
        textToSpeechSleep("Đã xảy ra lỗi xử lý hình ảnh", 1.2, False)
        handleImage(camera, labelWidget, faceDetection, tracker)

def addNewEntry(code, name, image, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")

    global dataListGlobal
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
        faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        start = time.time()
        tracker = CentroidTracker(maxDisappeared=5, maxDistance=60, useCache=True, cacheLifetime=1.5, killJob=killJob)
        print(f'CentroidTracker: {time.time() - start}')
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
                handleImage(camera, labelWidget, faceDetection, tracker)

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