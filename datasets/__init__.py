from .ucm import UCM_Dataset

import random
import torch
import numpy as np

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from torchvision import transforms as T

# normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
#                        std = [0.229, 0.224, 0.225])


# no need to normalize
train_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor()])

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])

def get_dataloader(mode, dataset_name, dataset_path, batchsize, num_workers):
    if dataset_name == 'ucm':
        if mode == 'train':
            data = UCM_Dataset(mode, img_dir=dataset_path, transform = train_transform)
            data_loader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=num_workers,
                                                      pin_memory=False, drop_last=False)
        elif mode =='val' or mode == 'test' or mode == 'attack':
            data = UCM_Dataset(mode, img_dir=dataset_path, transform = test_transform)
            data_loader = torch.utils.data.DataLoader(data, batch_size=batchsize, shuffle=False,
                                                      num_workers=num_workers, pin_memory=False, drop_last=False)

    data_loader.name = dataset_name
    data_loader.batch = batchsize
    return data_loader


# check if it works
# labels = []
# dataloader = get_dataloader(mode='train', dataset_name='ucm', dataset_path='E:/RS_adv/data/UCMerced_LandUse/Images', batchsize=32, num_workers=0)
# for i, [img, label, target_label] in enumerate(dataloader):
#     labels.extend(label.numpy().tolist())
#
# dataloader = get_dataloader(mode='val', dataset_name='ucm', dataset_path='E:/RS_adv/data/UCMerced_LandUse/Images', batchsize=32, num_workers=0)
# for i, [img, label, target_label] in enumerate(dataloader):
#     labels.extend(label.numpy().tolist())
#
# from collections import Counter
# print(Counter(labels))