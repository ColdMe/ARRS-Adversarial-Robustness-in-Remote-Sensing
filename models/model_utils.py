from torchvision import models
from torch import nn
import torch
import timm

def get_model(model_name, dataset_name, device='cpu', ckpt_path=None):
    num_classes = 0
    
    # set num_classes for different datasets
    if dataset_name == 'ucm':
        num_classes = 21
    elif dataset_name == 'aid':
        num_classes = 30
    elif dataset_name == 'mstar':
        num_classes = 8
        
    # build the models and revise the classifier
    if model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
        if dataset_name == 'mstar':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_name == 'inception':
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'vit':
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        model.reset_classifier(num_classes)
    elif model_name == 'deit':
        model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        model.reset_classifier(num_classes)
    elif model_name == 'swin':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        model.reset_classifier(num_classes)
        
    # retrieve the checkpoints if any
    if ckpt_path:
        pretrain_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(pretrain_dict)

    model = model.to(device)
    return model