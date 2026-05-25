import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.nn.functional as F
import random
import os
import numpy as np
import logging
import math
from models.quant_function import ReScaWConv
# from timm.models import resume_checkpoint

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def proposedLoss(outputs, fire_rate, labels, criterion, lamb):
    T = outputs.size(1)
    Loss_es, Loss_f = 0, 0
    for t in range(T):
        Loss_es += criterion(outputs[:,t,:], labels)
    Loss_es = Loss_es / T
    if lamb != 0:
        Loss_f = (sum([Entro(ele) for ele in fire_rate[1:-1]])) / len(fire_rate[1:-1])  # del the first and the last layer
    return Loss_es + lamb * Loss_f # L_Total

def Entro(rate):
    return (rate - 0.5) ** 2

def Log_UP(K_min, K_max, Epochs, epoch):
    Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
    return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / Epochs * epoch)]).float().cuda()

def res18KT(model, k, t):
    for i in range(2):  # iter of the basicblock
        model.layer1[i].conv1.k = k
        model.layer1[i].conv2.k = k
        model.layer1[i].conv1.t = t
        model.layer1[i].conv2.t = t

        model.layer2[i].conv1.k = k
        model.layer2[i].conv2.k = k
        model.layer2[i].conv1.t = t
        model.layer2[i].conv2.t = t

        model.layer3[i].conv1.k = k
        model.layer3[i].conv2.k = k
        model.layer3[i].conv1.t = t
        model.layer3[i].conv2.t = t

        model.layer4[i].conv1.k = k
        model.layer4[i].conv2.k = k
        model.layer4[i].conv1.t = t
        model.layer4[i].conv2.t = t

    return model

def res19KT(model, k, t):

    for i in range(3):  # iter of the basicblock
        model.layer1[i].conv1.k = k
        model.layer1[i].conv2.k = k
        model.layer1[i].conv1.t = t
        model.layer1[i].conv2.t = t

        model.layer2[i].conv1.k = k
        model.layer2[i].conv2.k = k
        model.layer2[i].conv1.t = t
        model.layer2[i].conv2.t = t

        # model.layer1[i].conv1.module.k = k
        # model.layer1[i].conv2.module.k = k
        # model.layer1[i].conv1.module.t = t
        # model.layer1[i].conv2.module.t = t
        #
        # model.layer2[i].conv1.module.k = k
        # model.layer2[i].conv2.module.k = k
        # model.layer2[i].conv1.module.t = t
        # model.layer2[i].conv2.module.t = t

        if i < 2:
            model.layer3[i].conv1.k = k
            model.layer3[i].conv2.k = k
            model.layer3[i].conv1.t = t
            model.layer3[i].conv2.t = t
            # model.layer3[i].conv1.module.k = k
            # model.layer3[i].conv2.module.k = k
            # model.layer3[i].conv1.module.t = t
            # model.layer3[i].conv2.module.t = t

    return model


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:,t,:], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

def save_checkpoint(state, save, dirname):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, dirname)
    torch.save(state, filename)


# def split_weights(model):
#     """split network weights into to categlories,
#     one are weights in conv layer and linear layer,
#     others are other learnable paramters(conv bias,
#     bn weights, bn bias, linear bias)
#
#     Args:
#         net: network architecture
#
#     Returns:
#         a dictionary of params splite into to categlories
#     """
#     all_parameters = model.parameters()
#     weight_parameters = []
#
#     for pname, p in model.named_parameters():
#         if p.ndimension() == 4 or 'conv' in pname:
#             weight_parameters.append(p)
#
#     weight_parameters_id = list(map(id, weight_parameters))
#     other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
#
#     return [dict(params=weight_parameters), dict(params=other_parameters, weight_decay=0)]



#

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias,
    bn weights, bn bias, linear bias)

    Args:
        net: network architecture

    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if hasattr(m, 'sign'):
                no_decay.append(m.sign)

            if m.bias is not None:
                no_decay.append(m.bias)

        elif isinstance(m, ReScaWConv):
            decay.append(m.weight)

            if hasattr(m, 'clip_val'):
                no_decay.append(m.clip_val)
            if hasattr(m, 'zero'):
                no_decay.append(m.zero)
        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
            if hasattr(m, 'clip_val'):
                no_decay.append(m.clip_val)
            # if hasattr(m, 'thresh'):
            #     no_decay.append(m.thresh)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]