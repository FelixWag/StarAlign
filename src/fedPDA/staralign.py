import logging
from collections import OrderedDict
import copy

from torch import nn
import torch.optim as optim

from src.ml_training.local_supervised_learning import train_one_step, apply_average_gradient_step
from src.utils import argumentlib
from src.utils.argumentlib import args
from tqdm import tqdm
import numpy as np

import operator
from numbers import Number


def staralign_train(model, dataloader, device, average_grads, beta, loss_fun=nn.CrossEntropyLoss()):
    # In practice, this is executed locally on each client

    # First copy the model state dict
    initial_model_state = copy.deepcopy(model.state_dict())
    # Set the model to train mode
    model.train()
    # Create optimizer
    staralign_optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # Create a generator in order to not iterate over the whole dataset
    generator = iter(dataloader)
    if average_grads is None:
        generator_2 = iter(dataloader)
    losses = []

    for _ in tqdm(range(args.client_number_iterations), leave=True, position=0):
        try:
            data_batch, target_batch = next(generator)
        except:
            logging.info('Created a new iterator')
            # If generator is exhausted, create a new one
            generator = iter(dataloader)
            data_batch, target_batch = next(generator)
        loss = train_one_step(data_batch, target_batch, model, staralign_optimizer, device, loss_fun=loss_fun)
        losses.append(loss.item())
        # --------------------------- StarAlign Interleaving Step ----------------
        if average_grads is None:
            # If it is the target client
            try:
                data_batch, target_batch = next(generator_2)
            except:
                logging.info('Created a new iterator for generator_2')
                # If generator is exhausted, create a new one
                generator_2 = iter(dataloader)
                data_batch, target_batch = next(generator_2)
            loss = train_one_step(data_batch, target_batch, model, staralign_optimizer, device, loss_fun=loss_fun)
            losses.append(loss.item())
        else:
            # If it is a source client
            apply_average_gradient_step(model=model, optimizer=staralign_optimizer, average_grads=average_grads)
        # ------------------------------------------------------------------------
    new_weights = staralign_update(initial_weights=initial_model_state, updated_weights=model.state_dict(), beta=beta)

    return np.array(losses).mean(), new_weights


class ParamDict(OrderedDict):
    # NOTE: This class implementation was taken from: https://github.com/YugeTen/fish
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def staralign_update(initial_weights, updated_weights, beta):
    new_weights, updated_weights = ParamDict(initial_weights), ParamDict(updated_weights)
    new_weights += beta * (updated_weights - new_weights)
    return new_weights
