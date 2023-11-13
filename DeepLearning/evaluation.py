import torch
def accuracy(labels_pred, labels):

    _, predicted = labels_pred.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct