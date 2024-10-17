import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

datapath="./data/MICCAI-BraTS2024-MET-Challenge-Training_1/"

img_folders = [x[0].replace(datapath, "") for x in os.walk(datapath)]

for folder in tqdm(img_folders[1:]):
    folder_path = os.path.join(datapath, folder)

    t1c = os.path.join(folder_path, folder + "-t1c.nii.gz") 
    seg = os.path.join(folder_path, folder + "-seg.nii.gz")

    img = nib.load(t1c)
    seg = nib.load(seg)
    
    img = img.get_fdata()
    img = (img/img.max())*255
    img = np.transpose(img, (2,0,1))
    seg = seg.get_fdata()
    seg = seg*50
    seg = np.transpose(seg, (2,0,1))

    # print(img.shape, seg.shape)

    for i in range(img.shape[0]):
        cv2.imwrite('./data/2D_Train/img/' + folder + '-' + str(i) +'.jpg', img[i])
        cv2.imwrite('./data/2D_Train/label/' + folder + '-' + str(i) +'.jpg', seg[i])




