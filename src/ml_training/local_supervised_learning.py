import numpy as np
import torch
import torch.optim

import statistics
from tqdm import tqdm
from torch import nn
import logging
import copy

from collections import OrderedDict
from src.utils.argumentlib import args as arguments


def compute_average_learning_direction(func):
    def wrapper(*args, **kwargs):
        grad_tracker = GradientTracker(number_local_steps=arguments.client_number_iterations, **kwargs)
        loss = func(*args, **kwargs)
        grad_tracker.remove_hooks()
        avg_grads = grad_tracker.get_average_gradients()
        return loss, avg_grads

    return wrapper


def local_train(model, dataloader, optimizer, device, loss_fun=nn.CrossEntropyLoss()):
    model.train()
    iterations_loss = []

    # Create a generator in order to not iterate over the whole dataset
    generator = iter(dataloader)
    for _ in tqdm(range(arguments.client_number_iterations), leave=True, position=0):
        try:
            data_batch, target_batch = next(generator)
        except:
            print('[INFO] created a new iterator')
            # If generator is exhausted, create a new one
            generator = iter(dataloader)
            data_batch, target_batch = next(generator)
        loss = train_one_step(data_batch, target_batch, model, optimizer, device, loss_fun=loss_fun)
        iterations_loss.append(loss.item())
    return np.array(iterations_loss).mean()


def train_one_step(data_batch, target_batch, model, optimizer, device, loss_fun=nn.CrossEntropyLoss()):
    # Put images and labels onto the device
    data_batch, target_batch = data_batch.to(device), target_batch.to(device)

    # --------------------------- Forward Pass -> ---------------------------
    outputs = model(data_batch)
    target_batch = torch.argmax(target_batch, dim=1).long()
    # Do not add softmax to outputs since CrossEntropy loss already does it! outputs = F.softmax(outputs, dim=1)
    loss = loss_fun(outputs, target_batch)
    # ------------------------------------------------------------------------

    # --------------------------- Backward Pass <- ---------------------------
    # Zero the gradients
    optimizer.zero_grad()
    loss.backward()
    # Step with the optimizer
    optimizer.step()
    # ------------------------------------------------------------------------

    return loss


class GradientTracker:
    """
    This class is used to track the gradients of a model. It is used to compute the average gradient direction.
    """

    def __init__(self, model, number_local_steps, **_):
        self.gradient_sum = OrderedDict()
        self.number_local_steps = number_local_steps
        self.hooks = []
        self.add_hooks(model=model)

    def add_hooks(self, model):
        for name, param in model.named_parameters():
            handle = param.register_hook(self.gradient_adder(name))
            self.hooks.append(handle)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_gradient_sum(self):
        self.gradient_sum = OrderedDict()

    def gradient_adder(self, name):
        def hook(grad):
            grad = copy.deepcopy(grad)
            if name not in self.gradient_sum:
                self.gradient_sum[name] = grad
            else:
                self.gradient_sum[name] += grad

        return hook

    def get_average_gradients(self):
        return {k: v / self.number_local_steps for k, v in self.gradient_sum.items()}


def apply_average_gradient_step(model, optimizer, average_grads):
    # First set the gradients of the model to zero
    model.zero_grad()
    # Apply the gradient to all the gradients in the model
    for name, param in model.named_parameters():
        param.grad = average_grads[name]

    # Now make a gradient step
    optimizer.step()
