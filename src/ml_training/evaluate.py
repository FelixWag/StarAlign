import torch
import numpy as np
import logging

from torch import nn


def _construct_confusion_matrix(confusion_matrix, preds, gt):
    for t, p in zip(gt.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1


def format_per_class_accuracy(confusion_matrix):
    pre_class_accuracies_list = (confusion_matrix.diag() / confusion_matrix.sum(1)).tolist()
    for class_idx, accuracy in enumerate(pre_class_accuracies_list):
        logging.info(f"Per-Class: Class {class_idx}: Accuracy = {accuracy:.2%}")
    # Log overall accuracy
    logging.info('-' * 50)
    logging.info(f"Overall accuracy: {(confusion_matrix.diag().sum() / confusion_matrix.sum()):.2%}")


def evaluate_model(model, dataloader, device, num_classes, loss_fun=nn.CrossEntropyLoss()):
    model.eval()
    valid_loss = []
    total = 0
    correct = 0

    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            out = model(data)

            label = torch.argmax(label, dim=1).long()

            # Compute the loss
            loss = loss_fun(out, label)
            valid_loss.append(loss.item())

            # Evaluate accuracy
            total += label.size(0)
            # out = F.softmax(out, dim=1)

            # Max(1) returned maximum value for each tensor + [1] selects the indices
            pred = out.max(1)[1]
            _construct_confusion_matrix(confusion_matrix=confusion_matrix, gt=label, preds=pred)
            correct += pred.eq(label.view(-1)).sum().item()

    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    weighted_accuracy = torch.mean(per_class_accuracy)

    format_per_class_accuracy(confusion_matrix=confusion_matrix)
    logging.info(f"Weighted accuracy: {weighted_accuracy.item():.2%}")

    return np.array(valid_loss).mean(), correct, total, weighted_accuracy.item()
