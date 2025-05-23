
def getFaceLandmarks(frame, resultsMesh, resultsDetection, faceMesh, displayText, Color):
    hFrame, wFrame, _ = frame.shape
    kpss, bboxes, rects = [], [], []

    if resultsMesh.multi_face_landmarks:
        for i, faceLandmarks in enumerate(resultsMesh.multi_face_landmarks):
            lm = faceLandmarks.landmark
            kps = np.array([
                [lm[468].x * wFrame, lm[468].y * hFrame],
                [lm[473].x * wFrame, lm[473].y * hFrame],
                [lm[4].x * wFrame, lm[4].y * hFrame],
                [lm[61].x * wFrame, lm[61].y * hFrame],
                [lm[291].x * wFrame, lm[291].y * hFrame],
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
            faceWidth = x2New - x1New
            print(f'faceWidth: {faceWidth}')

            if faceWidth >= 20:
                rects.append((x1New, y1New, x2New - x1New, y2New - y1New, i))
                if 20 <= faceWidth < 65:
                    frame = displayText(frame, x1New, y1New, faceWidth, True, True, True, True, False, Color)

    elif resultsDetection.detections:
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
                    faceWidth = x2New - x1New
                    print(f'faceWidth (detection): {faceWidth}')

                    if faceWidth >= 20:
                        rects.append((x1New, y1New, x2New - x1New, y2New - y1New, i))
                        if 20 <= faceWidth < 65:
                            frame = displayText(frame, x1New, y1New, faceWidth, True, True, True, True, False, Color)

    return frame, kpss, bboxes, rects

frame, kpss, bboxes, rects = getFaceLandmarks(
    frame,
    resultsMesh=results, 
    resultsDetection=resultsDetection, 
    faceMesh=faceMesh,
    displayText=displayText,
    Color=Color
)
Dùng để kiển tra nên dùng loại phát hiện nào