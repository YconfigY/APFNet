import torch.optim as optim

from utils.config import cfg
from APFNet.utils.model_params import get_all_params, get_learnable_params


def set_optimizer(model, lr_base, lr_layer, lr_mult, train_all=False):
    if train_all:
        params = get_all_params(model)
    else:
        params = get_learnable_params(model)
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for layer, mult in zip(lr_layer, lr_mult):
            if k.startswith(layer):
                lr = lr_base * mult
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM, 
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    return optimizer