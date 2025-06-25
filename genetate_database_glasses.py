import os 
import sys
from tqdm import tqdm
import numpy as np
import cv2
import random
from create_mask.mask import create_mask
from create_glasses.glasses import create_glasses



glasses_list = []


default_path = os.path.join(os.getcwd(),"create_glasses/sun_glasses.png")
white_path = os.path.join(os.getcwd(),"create_glasses/glasses_white.png")
black_path = os.path.join(os.getcwd(),"create_glasses/glasses_black.png")



glasses_list.append(default_path)
glasses_list.append(white_path)
glasses_list.append(black_path)


dataset_path = 'dataset'


if not os.path.exists('dataset_with_glasses'):
    os.mkdir('dataset_with_glasses')

for i in os.listdir('dataset'):
    if not os.path.exists(f'dataset_with_glasses/{i}'):
        os.mkdir(f'dataset_with_glasses/{i}')        
        

        
imagePaths = []


for i in os.listdir(dataset_path):
    for j in os.listdir(f'{dataset_path}/{i}'):
        imagePaths.append(f'{dataset_path}/{i}/{j}')
        


for i in tqdm(imagePaths,total=len(imagePaths)):
    glasses_path = random.choice(glasses_list)
    create_glasses(i,glasses_path)


imagePaths = []



for i in os.listdir("dataset_with_glasses"):
    dir_path = f'dataset_with_glasses/{i}'
    if os.path.isdir(dir_path):
        for j in os.listdir(dir_path):
            imagePaths.append(f'{dir_path}/{j}')



# if not os.path.exists('dataset_with_mask_face'):
#     os.mkdir('dataset_with_mask_face')

#     for i in os.listdir("dataset_with_mask"):
#         os.mkdir(f'dataset_with_mask_face/{i}')


print('MaskAppending Done')
print("extract_faces")

protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


for (i,imagePath) in tqdm(enumerate(imagePaths),total=len(imagePaths)):
    # print(imagePath)
    face_path = imagePath.replace('dataset_with_mask','dataset')
    # print(face_path)
    image = cv2.imread(imagePath)
    face_image = cv2.imread(face_path)
    if image is None or face_image is None:
        print(f"Cannot read image: {imagePath} or {face_path}")
        if os.path.exists(imagePath):
            os.remove(imagePath)
        continue
    (h,w) = image.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(face_image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)


    detector.setInput(imageBlob)
    detections = detector.forward()
    # write_path = imagePath.replace('dataset_with_mask','dataset_with_mask_face')

    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > 0.5:

            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")

            face = image[startY:endY,startX:endX]
            try:
                face = cv2.resize(face,(224,224))
                cv2.imwrite(imagePath,face)
            except:
                print(f"Some error occured in {imagePath}")
                os.remove(imagePath)
        else:
            os.remove(imagePath)
