import torch.nn as nn
from collections import OrderedDict
import torchvision
import logging


def get_normalization_layer_names(state_dict, model):
    # For each new architecture we need to specify the corresponding layer names
    if isinstance(model, torchvision.models.densenet.DenseNet):
        return [k for k in list(state_dict.keys()) if 'norm' in k]
    else:
        raise ValueError(f"{type(model)} is not supported")


def reset_bn_running_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def set_bn_to_train_disable_grad(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.requires_grad_(False)


def estimate_bn_stats(model, dataloader, device, n_iter=20):
    logging.info("Estimating fedBN statistics on unseen client")
    # Set model to eval
    model.eval()
    # Reset BN params
    reset_bn_running_stats(model)
    # set BN params to train + learnables requires to false
    set_bn_to_train_disable_grad(model)
    # Iterate over model k times
    for i in range(n_iter):
        for data, _ in dataloader:
            data = data.to(device)
            model(data)
