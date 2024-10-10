import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# test_file = os.path.join("./data/MICCAI-BraTS2024-MET-Challenge-Training_1/BraTS-MET-00001-000/", "BraTS-MET-00001-000-t1c.nii.gz")

# img = nib.load(test_file)
# data = img.get_fdata()

# print(data.shape)


class BratsDataset(Dataset):
    def __init__(self, datapath="./data/MICCAI-BraTS2024-MET-Challenge-Training_1/"):
        self.datapath = datapath
        self.img_folders = [x[0].replace(datapath, "") for x in os.walk(datapath)]

    def __len__(self):
        return len(self.img_folders) - 1
    
    def __getitem__(self, index):
        img_folder = os.path.join(self.datapath, self.img_folders[index])
        print(index)

        img = nib.load(os.path.join(img_folder, self.img_folders[index] + "-t1c.nii.gz"))
        img = img.get_fdata(dtype=np.float32)

        label = nib.load(os.path.join(img_folder, self.img_folders[index] + "-seg.nii.gz"))
        label = label.get_fdata(dtype=np.float32)

        return np.expand_dims(img, 0), np.expand_dims(label,0)
    