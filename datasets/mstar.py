import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random


class MSTAR_Dataset(Dataset):
    '''
    This is a 8 class image dataset meant for research purposes.
    There are 1000 images for each class.
    Each image measures 368x368 pixels.
    '''
    def __init__(self, mode, img_dir, ratio=0.8, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = os.listdir(self.img_dir)
        self.img_names = []
        self.img_paths = []
        self.labels = []
        for i, class_name in enumerate(self.classes):
            filefolder = os.path.join(self.img_dir, class_name)
            for img_name in os.listdir(filefolder):
                if img_name.endswith('.JPG'):
                    self.img_names.append(img_name)
                    self.img_paths.append(os.path.join(filefolder, img_name))
                    self.labels.append(i)

        # set seed and shuffle them!
        # because of the random seed, don't worry about the split.
        random.seed(0)
        li = list(zip(self.img_names, self.img_paths, self.labels))
        random.shuffle(li)
        self.img_names, self.img_paths, self.labels = zip(*li)
        self.img_names, self.img_paths, self.labels = list(self.img_names), list(self.img_paths), list(self.labels)
        num = len(self.labels)
        if mode == 'train':
            self.img_names = self.img_names[:int(ratio*num)]
            self.img_paths = self.img_paths[:int(ratio*num)]
            self.labels = self.labels[:int(ratio*num)]
        elif mode == 'test' or mode == 'val':
            self.img_names = self.img_names[int(ratio*num):]
            self.img_paths = self.img_paths[int(ratio*num):]
            self.labels = self.labels[int(ratio*num):]
        elif mode == 'attack':
            pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        image_path = self.img_paths[idx]
        label = self.labels[idx]
        image = np.array(Image.open(image_path).convert('L'))
        image = Image.fromarray(image)
        if self.transform:
            image=self.transform(image)
        return image, int(label), int(label)

# mstar = MSTAR_Dataset(mode='attack', img_dir='/root/autodl-tmp/MSTAR')
# print(mstar.__getitem__(0))