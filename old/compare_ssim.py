from skimage.metrics import structural_similarity as ssim
import cv2

def compareSsim(image1Path, image2Path):

    img1 = cv2.imread(image1Path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2Path, cv2.IMREAD_GRAYSCALE)

    similarityIndex, _ = ssim(img1, img2, full=True)

    print(f"SSIM: {similarityIndex}")

    if similarityIndex > 0.9: 
        print("Hai bức ảnh rất giống nhau!")
        return True
    else:
        print("Hai bức ảnh khác nhau.")
        return False

image1 = "stored-faces/001/face_2025-01-22_15-53-51_310.jpg"
image2 = "stored-faces/001/face_2025-01-22_15-53-52_651.jpg"
compareSsim(image1, image2)
