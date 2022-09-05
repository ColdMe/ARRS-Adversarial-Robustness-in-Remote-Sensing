import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import random


class UCM_Dataset(Dataset):
    '''
    This is a 21 class land use image dataset meant for research purposes.
    There are 100 images for each class.
    Each image measures 256x256 pixels.
    The images were manually extracted from large images from the USGS National Map Urban Area Imagery collection for various urban areas around the country. The pixel resolution of this public domain imagery is 1 foot.
    Please cite the following paper when publishing results that use this dataset:
    Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.
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
            self.img_names.extend(os.listdir(filefolder))
            self.img_paths.extend([os.path.join(filefolder, p) for p in os.listdir(filefolder)])
            self.labels.extend([i] * len(os.listdir(filefolder)))

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
        image = np.array(Image.open(image_path).convert('RGB'))
        image = Image.fromarray(image)
        if self.transform:
            image=self.transform(image)
        return image, int(label), int(label)

# ucm =UCM_Dataset('E:/RS_adv/data/UCMerced_LandUse/Images',)
# ucm.__getitem__(0)