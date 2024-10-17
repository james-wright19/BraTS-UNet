import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

# test_file = os.path.join("./data/MICCAI-BraTS2024-MET-Challenge-Training_1/BraTS-MET-00001-000/", "BraTS-MET-00001-000-t1c.nii.gz")

# img = nib.load(test_file)
# data = img.get_fdata()

# print(data.shape)

resize = Resize((64,64))

class BratsDataset(Dataset):
    def __init__(self, datapath="./data/2D_Train"):
        self.datapath = datapath
        self.img = os.listdir(os.path.join(datapath, 'img'))

    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.datapath, 'img', self.img[index])
        label_path = os.path.join(self.datapath, 'label', self.img[index])

        image = read_image(img_path)
        label = read_image(label_path)

        image = resize(image)
        label = resize(label)
        
        return image, label.squeeze(0)
    