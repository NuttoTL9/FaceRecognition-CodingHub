import os 
import sys
from tqdm import tqdm
import numpy as np
import cv2
import random
from create_mask.mask import create_mask

# เตรียม list หน้ากาก
mask_list = [
    os.path.join(os.getcwd(), "mask/mask.png"),
    os.path.join(os.getcwd(), "mask/white.png"),
    os.path.join(os.getcwd(), "mask/blue.png"),
    os.path.join(os.getcwd(), "mask/black.png")
]

dataset_path = 'data'

# สร้างโฟลเดอร์เก็บภาพที่มี mask
if not os.path.exists('dataset_with_mask'):
    os.makedirs('dataset_with_mask')

# เก็บ path ภาพทั้งหมดจาก dataset_path
imagePaths = []
for i in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, i)
    if os.path.isfile(image_path):
        imagePaths.append(image_path)
    elif os.path.isdir(image_path):
        for j in os.listdir(image_path):
            sub_image_path = os.path.join(image_path, j)
            if os.path.isfile(sub_image_path):
                imagePaths.append(sub_image_path)

# สร้างโฟลเดอร์ย่อยให้ตรงกับภาพต้นทาง
for path in imagePaths:
    rel_path = os.path.relpath(path, dataset_path)  # ส่วน path ต่อจาก data/
    save_dir = os.path.join('dataset_with_mask', os.path.dirname(rel_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# วนใส่ mask ให้ภาพทั้งหมด
for path in tqdm(imagePaths, total=len(imagePaths)):
    mask_path = random.choice(mask_list)
    create_mask(path, mask_path)

print('Mask Appending Done')
print("Start Extract Faces...")

# เตรียมโมเดลตรวจจับใบหน้า
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# เก็บ path ภาพที่ใส่ mask แล้ว
imagePaths = []
for root, dirs, files in os.walk('dataset_with_mask'):
    for file in files:
        imagePaths.append(os.path.join(root, file))

# วนตรวจจับใบหน้า ตัดเฉพาะส่วนใบหน้าเก็บทับไฟล์เดิม
for imagePath in tqdm(imagePaths, total=len(imagePaths)):
    face_path = imagePath.replace('dataset_with_mask', 'data')
    
    image = cv2.imread(imagePath)
    face_image = cv2.imread(face_path)
    
    if image is None or face_image is None:
        print(f"Error reading image {imagePath} หรือ {face_path}")
        os.remove(imagePath)
        continue

    (h, w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(face_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    if detections.shape[2] > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (224, 224))
                cv2.imwrite(imagePath, face)
            except:
                print(f"Resize or Save Error in {imagePath}")
                os.remove(imagePath)
        else:
            os.remove(imagePath)
    else:
        os.remove(imagePath)

print("Face extraction completed.")
