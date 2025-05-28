import cv2
import numpy as np
import time
from facesController import CameraReader

cam1 = CameraReader(0)
cam2 = CameraReader(2)
cam1.start()
cam2.start()

frameCount = 0
mainFps = 0
startTime = time.time()

try:
    while True:
        frameCount += 1
        currentTime = time.time()
        elapsedTime = currentTime - startTime

        if elapsedTime >= 1.0:
            mainFps = frameCount / elapsedTime
            frameCount = 0
            startTime = currentTime

        frame1 = cam1.getFrame()
        frame2 = cam2.getFrame()

        if not cam1.getStatus():
            frame1 = cv2.putText(np.zeros((480, 640, 3), dtype=np.uint8), "Camera 1 Error", (50, 240),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            fps1 = cam1.getFps()
            cv2.putText(frame1, f'FPS: {fps1:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not cam2.getStatus():
            frame2 = cv2.putText(np.zeros((480, 640, 3), dtype=np.uint8), "Camera 2 Error", (50, 240),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            fps2 = cam2.getFps()
            cv2.putText(frame2, f'FPS: {fps2:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if frame1 is not None and frame2 is not None:
            combined = cv2.hconcat([frame1, frame2])
            cv2.putText(combined, f'Main FPS: {mainFps:.2f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Face Detection - Dual Camera", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam1.stop()
    cam2.stop()
    cam1.join()
    cam2.join()
    cv2.destroyAllWindows()
