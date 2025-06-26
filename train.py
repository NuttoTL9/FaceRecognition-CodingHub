import torch
import os
import numpy as np
from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from fastai.callback.all import ShowGraphCallback

# กำหนด device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# Path Dataset และ Model
dataset_path = 'dataset_with_mask'
model_dir = f'{dataset_path}/models'  # ให้โมเดลไปเก็บใน dataset_with_mask/models

np.random.seed(42)

# เตรียม Data
trfm = aug_transforms(do_flip=True, flip_vert=True, max_zoom=1.2, max_rotate=20.0, max_lighting=0.4)
data = ImageDataLoaders.from_folder(
    dataset_path,
    valid_pct=0.2,
    item_tfms=Resize(224),
    batch_tfms=trfm,
    num_workers=4
).cuda()

# สร้าง Learner พร้อมกำหนดตำแหน่งเก็บโมเดล
learn = cnn_learner(data, resnet34, metrics=[error_rate, accuracy], cbs=ShowGraphCallback, model_dir=model_dir)

# ---------------------
# TRAIN & SAVE
# ---------------------
learn.fit_one_cycle(10, lr_max=1e-3)
learn.save('stage-1')  # โมเดลจะถูกเซฟไว้ใน dataset_with_mask/models/stage-1.pth

# ---------------------
# LOAD โมเดลที่เทรนไว้ (หากต้องการ)
# ---------------------
learn.load('stage-1', with_opt=True, device=device)

# ---------------------
# INTERPRET & EVALUATE
# ---------------------
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# ---------------------
# EXPORT สำหรับใช้งานจริง
# ---------------------
learn.export()  # ไฟล์ export.pkl จะถูกเซฟไว้ใน dataset_with_mask/models
