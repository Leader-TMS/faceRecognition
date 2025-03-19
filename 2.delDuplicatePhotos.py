import os
import imagehash
from PIL import Image

def getImageHash(filePath):
    image = Image.open(filePath)
    return imagehash.average_hash(image)

def findDuplicateImagesInSubfolders(directory):
    duplicates = []
    
    for root, dirs, files in os.walk(directory):
        if root != directory:
            print(f'Checking folder: {root}')
            images = {}
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    filePath = os.path.join(root, file)
                    fileHash = getImageHash(filePath)
                    
                    if fileHash in images:
                        duplicates.append(filePath)
                    else:
                        images[fileHash] = filePath
    print(f'Total duplicates found: {len(duplicates)}')              
    for duplicate in duplicates:
    # print(f"Removing duplicate image: {duplicate}")
        os.remove(duplicate)

if __name__ == "__main__":
    directory = 'dataset'
    findDuplicateImagesInSubfolders(directory)

