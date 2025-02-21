# faceRecognition

<!-- lần 1
4 góc 240 -> 125 tấm: góc chính diện (nghiên vừa) 85% -> 96%, còn lại góc trái, phải, trên, dưới từ 60%
8 góc 320 -> 220 tấm: góc chính diện (nghiên vừa) 96% -> 99%, còn lại góc trái, phải, trên, dưới từ 85% -> 90%

lần 2 
4 góc 400 -> 100 tấm: góc chính diện (nghiên vừa) 93% -> 96%, còn lại góc trái, phải, trên, dưới từ 86%
8 góc 643 -> 192 tấm: góc chính diện (nghiên vừa) 93% -> 96%, còn lại góc trái, phải, trên, dưới từ 83% -> 88%

lần 3
4 góc 410 -> 121 tấm: góc chính diện (nghiên vừa) 94% -> 96%, còn lại góc trái, phải, trên, dưới từ 83%
8 góc 567 -> 222 tấm: góc chính diện (nghiên vừa) 96% -> 98%, còn lại góc trái, phải, trên, dưới từ 83% -> 98% -->

ArcFace:  Hệ thống an ninh, giám sát, nhận diện khuôn mặt trong đám đông.
VGGFace2: Nhận diện khuôn mặt đa dạng về chủng tộc, độ tuổi và điều kiện môi trường.

Datasets and Accuracy

Dataset Breakdown by Capture Angles and Performance:

Setup camera:

    VIDEO
    - Video type: Play videos
    - Resolution: 1280*720P
    - Bitrate type: Constant
    - Image Quality: Highest
    - Video frame rate(fps): 30
    - Maximum bit rate: 8192Kbps
    - Video encoding: H.264
    - H.264+: Off
    - File: Cao
    - I-Frame Interval: 150
    - SVC: On
    - Smooth: 50

    IMAGE
    - Adjust photos
        - Brightness: 50
        - Contrast: 49
        - Saturation: 55
        - Sharpness: 65

    - Contact settings
        - Iris mode: Permanent
        - Anti-striping: Off
        - Exposure time: 1/60
    
    - Day/night switch
        - Day/night switch: Auto
        - Sensitivity: 4
        - Filter time: 5
        - Add smart lighting: Off
        - Light supplement mode: Smart
        - Control the brightness of the light: Auto
        - IR light: 100
        - White light: 100

    - Backlighting
        - BLC Area: Off
        - WDR: Off
        - HLC: Off

    - White balance
        - White balance: Auto white balance 2/AWB2

    - Image Enhancement
        - Digital noise reduction: Expert mode
        - DNR level range: 50
        - DNR time level: 50
        - Cover mode: Auto
        - Gray Area: [0-255]

    - Customize Video
        - Mirror: Off
        - Turn: Off
        - Video Standard: NTSC(60HZ)


Round 1:

    4 angles (240 images):
        Front (slightly tilted): 85% → 96%
        Other angles (left, right, top, bottom): ~60%
    8 angles (320 images):
        Front (slightly tilted): 96% → 99%
        Other angles: 85% → 90%

Round 2:

    4 angles (400 images):
        Front (slightly tilted): 93% → 96%
        Other angles: ~86%
    8 angles (643 images):
        Front (slightly tilted): 93% → 96%
        Other angles: 83% → 88%

Round 3:

    4 angles (410 images):
        Front (slightly tilted): 94% → 96%
        Other angles: ~83%
    5 angles (567 images):
        Front (slightly tilted): 96% → 98%
        Other angles: 83% → 98%