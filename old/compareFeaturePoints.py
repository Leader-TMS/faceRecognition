import cv2
import os
import numpy as np

def detectFeatures(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    akaze = cv2.AKAZE_create()
    kp, des = akaze.detectAndCompute(gray, None)
    return kp, des

def compareImages(image1, image2):
    kp1, des1 = detectFeatures(image1)
    kp2, des2 = detectFeatures(image2)
    
    if des1 is None or des2 is None:
        return 0

    if des1.dtype != des2.dtype:
        if des1.dtype == np.float32 or des2.dtype == np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
        else:
            des1 = des1.astype(np.uint8)
            des2 = des2.astype(np.uint8)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def getBestMatchFolder(input_image, root_folder):
    max_matches = 0
    best_folder = "Unknown"
    
    for subdir, _, files in os.walk(root_folder):
        total_matches = 0
        
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path)
                matches = compareImages(input_image, image)
                total_matches += matches
        
        if total_matches > max_matches:
            max_matches = total_matches
            best_folder = subdir

    name_with_date = best_folder.split("/")[-1]
    name = name_with_date.split("_")[0]
    
    total_files = sum([len(files) for _, _, files in os.walk(root_folder)])
    match_percentage = (max_matches / total_files) * 100 if total_files > 0 else 0

    return name, max_matches, match_percentage

if __name__ == "__main__":
    input_image_path = 'input_image.jpg'
    input_image = cv2.imread(input_image_path)
    
    root_folder = 'images_folder'
    best_folder, match_count, match_percentage = getBestMatchFolder(input_image, root_folder)
    
    print(f"Thư mục con có sự tương đồng cao nhất: {best_folder} với {match_count} điểm trùng khớp.")
    print(f"Tỷ lệ trùng khớp: {match_percentage:.2f}%")
