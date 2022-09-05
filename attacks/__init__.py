from advertorch.attacks import LinfPGDAttack, LinfMomentumIterativeAttack, GradientSignAttack
from torch import nn

def get_attack(attack_name, model, eps=0.3, nb_iter=40):
    if attack_name == 'fgsm':
        adversary = GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, clip_min=0.0, clip_max=1.0, targeted=False)
    elif attack_name == 'pgd':
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=nb_iter, eps_iter=eps/nb_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif attack_name == 'mim':
        adversary = LinfMomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, nb_iter=nb_iter, decay_factor=1.0, eps_iter=eps/nb_iter, clip_min=0.0, clip_max=1.0, targeted=False)
    # elif attack_name == '':
        
    return adversary