import math

def calculateCenter(x, y, w, h):
    return x + w / 2, y + h / 2

def calculateScaleFactor(w1, h1, w2, h2):
    scaleX = w2 / w1 if w1 != 0 else 1
    scaleY = h2 / h1 if h1 != 0 else 1
    return scaleX, scaleY

def calculateVelocity2D(bbox1, bbox2, time):

    center1 = calculateCenter(bbox1[0], bbox1[1], bbox1[2], bbox1[3])
    center2 = calculateCenter(bbox2[0], bbox2[1], bbox2[2], bbox2[3])

    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    scaleX, scaleY = calculateScaleFactor(bbox1[2], bbox1[3], bbox2[2], bbox2[3])

    if time > 0:
        velocityMove = distance / time
    else:
        velocityMove = 0

    velocityScaleX = scaleX / time if time > 0 else 0
    velocityScaleY = scaleY / time if time > 0 else 0

    velocityTotal = math.sqrt(velocityMove**2 + velocityScaleX**2 + velocityScaleY**2)

    return velocityTotal, velocityMove, velocityScaleX, velocityScaleY

bbox1 = (10, 20, 50, 40)
bbox2 = (15, 25, 60, 50)
time = 0.5

velocityTotal, velocityMove, velocityScaleX, velocityScaleY = calculateVelocity2D(bbox1, bbox2, time)

print(f"Tốc độ tổng hợp: {velocityTotal} đơn vị/giây")
print(f"Tốc độ di chuyển: {velocityMove} đơn vị/giây")
print(f"Tốc độ thay đổi kích thước theo chiều rộng: {velocityScaleX} đơn vị/giây")
print(f"Tốc độ thay đổi kích thước theo chiều cao: {velocityScaleY} đơn vị/giây")