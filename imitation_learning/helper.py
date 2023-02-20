import torch
import random

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), axis=0)

def cross_entropy(y, y_pred):
    loss = -torch.sum(y*torch.log(y_pred))
    return loss/y_pred.shape[0]

def balance_dataset(data_df):
    unbalance_list = data_df.index[data_df[1] == 0.0].tolist()
    row_remove = random.choices(unbalance_list, k=int(len(unbalance_list)/1.25))
    data_df = data_df.drop(row_remove)

    return data_df