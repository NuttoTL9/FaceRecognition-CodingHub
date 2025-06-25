import os 
import sys
from tqdm import tqdm
import random
from glasses import create_glasses

glasses_list = []

default_path = os.path.join(os.getcwd(),"create_glasses/sun_glasses.png")
white_path = os.path.join(os.getcwd(),"create_glasses/glasses_white.png")
black_path = os.path.join(os.getcwd(),"create_glasses/glasses_black.png")
# black_path = os.path.join(os.getcwd(),"create_mask/black.png")

glasses_list.append(default_path)
glasses_list.append(white_path)
glasses_list.append(black_path)
# glasses_list.append(black_path)

print(glasses_list)

dataset_path = 'dataset'

if not os.path.exists('dataset_with_glasses'):
    os.mkdir('dataset_with_glasses')

for i in os.listdir('dataset'):
    if not os.path.exists(f'dataset_with_glasses/{i}'):
        os.mkdir(f'dataset_with_glasses/{i}')


imagePaths = []

for i in os.listdir('dataset'):
    for j in os.listdir(f'dataset/{i}'):
        imagePaths.append(f'dataset/{i}/{j}')


for i in tqdm(imagePaths,total=len(imagePaths)):
    glasses_path = random.choice(glasses_list)
    create_glasses(i,glasses_path)