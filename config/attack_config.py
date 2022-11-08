import numpy as np
import time

class AttackConfig:
    def __init__(self):
        # * dataset-------------------------
        # Dataset name, choices = ['ucm', 'aid', 'mstar']
        self.dataset_name = 'ucm'
        # dataset_path
        self.dataset_path = {'ucm': '/root/autodl-tmp/UCMerced_LandUse/Images',
                             'aid': '/root/autodl-tmp/AID',
                             'mstar': '/root/autodl-tmp/MSTAR',
                             }

        # model-------------------------
        # substitute model name, choices = ['resnet34', 'densenet', 'inception', 'vit', 'deit', 'swin']
        self.sub_model_name = 'resnet34'
        
        # substitute checkpoints path
        self.sub_ckpt_path = 'checkpoints/ucm_resnet34_bs64_lr0.001/ucm_resnet34_bs64_lr0.001_20220731_112927/epoch_100.pth'
        # self.sub_ckpt_path = 'checkpoints/ucm_resnet34_adv_bs64_lr0.01/ucm_resnet34_adv_bs64_lr0.01_20220816_150516/epoch_300.pth'
        # self.sub_ckpt_path = 'checkpoints/aid_resnet34_bs64_lr0.001/aid_resnet34_bs64_lr0.001_20220914_225046/epoch_100.pth'
        # self.sub_ckpt_path = 'checkpoints/mstar_resnet34_bs64_lr0.001/mstar_resnet34_bs64_lr0.001_20220915_202756/epoch_30.pth'
        # self.sub_ckpt_path = 'checkpoints/ucm_densenet_bs64_lr0.001/ucm_densenet_bs64_lr0.001_20220731_125048/epoch_100.pth'
        # self.sub_ckpt_path = 'checkpoints/ucm_vit_bs64_lr0.001/ucm_vit_bs64_lr0.001_20221008_124921/epoch_70.pth' # or 20
        
        # defense model name, choices = ['resnet34', 'densenet', 'inception', 'vit', 'deit', 'swin']
        self.defense_model_name = 'resnet34'
        
        # defense_checkpoints path
        self.defense_ckpt_path = 'checkpoints/ucm_resnet34_bs64_lr0.001/ucm_resnet34_bs64_lr0.001_20220731_112927/epoch_100.pth'
        # self.defense_ckpt_path = 'checkpoints/ucm_resnet34_adv_bs64_lr0.01/ucm_resnet34_adv_bs64_lr0.01_20220816_150516/epoch_300.pth'
        # self.defense_ckpt_path = 'checkpoints/aid_resnet34_bs64_lr0.001/aid_resnet34_bs64_lr0.001_20220914_225046/epoch_100.pth'
        # self.defense_ckpt_path = 'checkpoints/mstar_resnet34_bs64_lr0.001/mstar_resnet34_bs64_lr0.001_20220915_202756/epoch_30.pth'
        # self.defense_ckpt_path = 'checkpoints/ucm_densenet_bs64_lr0.001/ucm_densenet_bs64_lr0.001_20220731_125048/epoch_100.pth'

        
        # attack-------------------------
        # attack name, choices = ['', 'fgsm', 'pgd', 'mim']
        self.attack_name = 'fgsm'
        # batch size of the dataloader
        self.attack_batchsize = 16
        # You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2]
        self.norm = np.inf
        # epsilon, linf: 8 / 255.0 and l2: 3.0
        # self.eps = [8 / 255]
        # self.eps = [ i / 255.0 for i in range(5, 51, 5)]
        self.eps = [ i / 255.0 for i in range(51)]
        # number of iterations
        # self.nb_iter = range(1, 11)
        self.nb_iter = [5]
        # target or nontarget
        self.target = False
        # # loss for fgsm, bim, pgd, mim, dim and tim', choices = ['ce', 'cw']
        # self.loss = 'cw'
        
        # defense------------------------
        # defense name, choices = ['jpeg-filter', 'bits-squeezing', 'median-filter']
        # pass a empty list [] if no defenses applied
        self.defenses = []
        
        # output-------------------------
        # output file
        self.out_attack_folder = 'attack_outputs'

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