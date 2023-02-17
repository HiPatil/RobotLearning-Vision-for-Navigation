import torch

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), axis=0)

def cross_entropy(y, y_pred):
    loss = -torch.sum(y*torch.log(y_pred))
    return loss/y_pred.shape[0]