import os
import imagehash
from PIL import Image

def get_image_hash(file_path):
    image = Image.open(file_path)
    return imagehash.average_hash(image)

def find_duplicate_images_in_subfolders(directory):
    duplicates = []
    
    for root, dirs, files in os.walk(directory):
        if root != directory:
            print(f'Checking folder: {root}')
            images = {}
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(root, file)
                    file_hash = get_image_hash(file_path)
                    
                    if file_hash in images:
                        duplicates.append(file_path)
                    else:
                        images[file_hash] = file_path
    print(f'Total duplicates found: {len(duplicates)}')              
    return duplicates

directory = 'dataset'

duplicates = find_duplicate_images_in_subfolders(directory)

for duplicate in duplicates:
    # print(f"Removing duplicate image: {duplicate}")
    os.remove(duplicate)
