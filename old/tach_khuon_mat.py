import cv2
import numpy as np
import os
import uuid
import threading
import imagehash
from PIL import Image
import faces
import tensorflow as tf
import json

model = tf.keras.models.load_model("face_recognition_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)
if not os.path.exists('detected_persons'):
    os.makedirs('detected_persons')

with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
counter = 0
frame_array = []
threads = []
counter_thread = 0
max_threads = 2
semaphore = threading.Semaphore(max_threads)

def get_frame_hash(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return imagehash.phash(pil_image)

def remove_duplicates(frames):
    seen_hashes = set()
    unique_frames = []
    for frame in frames:
        frame_hash = get_frame_hash(frame)
        if frame_hash not in seen_hashes:
            seen_hashes.add(frame_hash)
            unique_frames.append(frame)
    return unique_frames

def get_frame_difference(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return np.sum(diff_thresh)

def remove_duplicates_by_diff(frames, threshold=100000):
    unique_frames = []
    last_frame = None
    for frame in frames:
        if last_frame is None:
            unique_frames.append(frame)
            last_frame = frame
        else:
            diff = get_frame_difference(last_frame, frame)
            if diff > threshold:
                unique_frames.append(frame)
            last_frame = frame
    return unique_frames

def detect_person(frame_array, counter_thread):
    with semaphore:
        try:
            unique_images = remove_duplicates(remove_duplicates_by_diff(frame_array))
            print(f'[{counter_thread}] frame_array: {len(frame_array)} - unique_images: {len(unique_images)}')
            print(f'[{counter_thread}] detect_person started')
            person_array = []
            yolo_net = cv2.dnn.readNet("yolo/yolov4.weights", "yolo/yolov4.cfg")
            layer_names = yolo_net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
            for index, frame in enumerate(unique_images):
                print(f'[{counter_thread}] - index: {index}')
                if isinstance(frame, np.ndarray):
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    if blob is None or blob.size == 0:
                        continue
                    yolo_net.setInput(blob)
                    outputs = yolo_net.forward(output_layers)
                    height, width, channels = frame.shape
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outputs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5 and class_id == 0:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    if len(indexes) > 0:
                        print(f'[{counter_thread}] Found {len(indexes)} person(s).')
                        for i in indexes.flatten():
                            if i < len(boxes):
                                x, y, w, h = boxes[i]
                                if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
                                    cropped_person = frame[y:y + h, x:x + w]
                                    if cropped_person.size > 0:
                                        # Nhận diện là ai
                                        # gray = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2GRAY)

                                        # img_resized = cv2.resize(gray, (224, 224))
                                        # img_resized = img_resized.reshape(1, 224, 224, 1).astype('float32') / 255.0

                                        # prediction = model.predict(img_resized)
                                        # label = np.argmax(prediction)

                                        # predicted_label = label_map[str(label)]

                                        # print(f"Predicted label-----------------------------: {predicted_label}")
                                        random_name = str(uuid.uuid4()) + ".jpg"
                                        cv2.imwrite(os.path.join('detected_persons', random_name), cropped_person)
                                        person_array.append(cropped_person)
                                    else:
                                        print(f"[{counter_thread}] Warning: Cropped person is empty!")
                            else:
                                print(f"[{counter_thread}] Invalid index {i}, skipping.")
            faces.detectedFace(person_array)
            print(f'[{counter_thread}] DONE')
        except Exception as e:
            print(f"[{counter_thread}] Error in processing batch: {e}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_array.append(frame)
    if (counter + 1) % fps == 0:
        try:
            thread = threading.Thread(target=detect_person, args=(frame_array.copy(), counter_thread,))
            thread.start()
            threads.append(thread)
            frame_array.clear()
            counter_thread += 1
        except Exception as e:
            print(f"Error in starting thread: {e}")
            frame_array.clear()

    counter += 1

    cv2.imshow('Person Detection', frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

for thread in threads:
    if thread is not None:
        print('Thread Done')
        thread.join()
    else:
        print("Thread is None, skipping join.")

cap.release()
cv2.destroyAllWindows()
