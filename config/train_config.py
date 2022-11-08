import numpy as np
import time

class TrainConfig:
    def __init__(self):
        # * dataset-------------------------
        # Dataset name, choices = ['ucm', 'aid', 'mstar']
        self.dataset_name = 'aid'
        # dataset_path
        self.dataset_path = {'ucm': '/root/autodl-tmp/UCMerced_LandUse/Images',
                             'aid': '/root/autodl-tmp/AID',
                             'mstar': '/root/autodl-tmp/MSTAR',
                             }

        # model-------------------------
        # model name, choices = ['resnet34', 'densenet', 'inception', 'vit', 'deit', 'swin']
        self.model_name = 'resnet34'
        # retrieve checkpoints and continue to train, set as "" if train from scratch
        # self.ckpt_path = 'checkpoints/ucm_resnet34_bs64_lr0.001/ucm_resnet34_bs64_lr0.001_20220731_112927/epoch_100.pth'
        self.ckpt_path = ''

        # train-------------------------
        # epoch
        self.train_batchsize = 64
        self.epochs = 100
        self.momentum = 0.9
        self.weight_decay=1e-2
        self.lr = 1e-3
        self.lr_period = 30
        self.lr_decay = 1e-1
        
        # output file
        self.out_ckpt_folder = 'checkpoints'
        
        # adversarial training-------------------------
        # whether adversarial training is applied
        self.adv_train = False
        # attack name, choices = ['', 'fgsm', 'pgd', 'mim']
        self.attack_name = 'fgsm'
        # You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2]
        self.norm = np.inf
        # epsilon, linf: 8 / 255.0 and l2: 3.0, only self.eps[0] is valid
        self.eps = [8 / 255]
        # number of iterations, only self.nb_iter[0] is valid
        self.nb_iter = [5]
        # target or nontarget
        self.target = False
        # # loss for fgsm, bim, pgd, mim, dim and tim', choices = ['ce', 'cw']

        # others-------------------------
        # Comma separated list of GPU ids
        self.gpu = '0'
        self.seed = 0
        self.num_workers = 0
       
    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
    
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warning.warn("Warning: config has not attribute %s" % k)
            setattr(self, k, v)