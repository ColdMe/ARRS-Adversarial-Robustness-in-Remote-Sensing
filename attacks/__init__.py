from advertorch.attacks import LinfPGDAttack, LinfMomentumIterativeAttack, GradientSignAttack, DeepfoolLinfAttack
from .di import DiversityInputLinf
from .fgsm import FGSM
from torch import nn

def get_attack(attack_name, model, eps=0.3, nb_iter=40):
    if attack_name == 'fgsm':
        adversary = GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, clip_min=0.0, clip_max=1.0, targeted=False)
    elif attack_name == 'fgsm2':
        adversary = FGSM(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, clip_min=0.0, clip_max=1.0, targeted=False)
    elif attack_name == 'pgd':
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=nb_iter, eps_iter=eps/nb_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif attack_name == 'mim':
        adversary = LinfMomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, nb_iter=nb_iter, decay_factor=1.0, eps_iter=eps/nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
    elif attack_name == 'deepfool':
        adversary = DeepfoolLinfAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, nb_iter=nb_iter, overshoot=0.02, clip_min=0.0, clip_max=1.0, targeted=False)
    elif attack_name == 'di':
        adversary = DiversityInputLinf(model, eps=eps, stepsize=eps/nb_iter, steps=nb_iter, decay=1.0, resize_rate=0.85, diversity_prob=0.7, loss='ce')
    return adversary