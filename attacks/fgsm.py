import torch.nn as nn

from .utils import clamp
from .utils import normalize_by_pnorm
from .utils import batch_multiply

from .base import Attack
from .base import LabelMixin


class FGSM(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(FGSM, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        self.nb_iter = 1
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y = None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

#         x, y = self._verify_and_process_inputs(x, y)
#         xadv = x.requires_grad_()
#         outputs = self.predict(xadv)

#         loss = self.loss_fn(outputs, y)
#         if self.targeted:
#             loss = -loss
#         loss.backward()
#         grad_sign = xadv.grad.detach().sign()

#         xadv = xadv + batch_multiply(self.eps, grad_sign)

#         xadv = clamp(xadv, self.clip_min, self.clip_max)

#           return xadv.detach()
    
    
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        
        grad = xadv.grad.detach()
        grad_sign = grad.sign()
        
        xadv1 = xadv + batch_multiply(self.eps, grad)
        outputs1 = self.predict(xadv1)
        loss1 = self.loss_fn(outputs1, y)

        xadv3 = xadv + batch_multiply(self.eps, grad_sign)
        xadv3 = clamp(xadv3, self.clip_min, self.clip_max)
        outputs3 = self.predict(xadv3)
        loss3 = self.loss_fn(outputs3, y)
        
        print("{} {} {}".format(self.eps, loss1, loss3))
        return xadv3.detach()