import time
import random
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

from network import ClassificationNetwork
from dataset import get_dataloader


def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    nr_epochs = 100
    batch_size = 64
    nr_of_classes = 9  # needs to be changed
    gpu = torch.device('cuda')

    infer_action = ClassificationNetwork(nr_of_classes)
    infer_action = infer_action.to(gpu)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-2)
    

    start_time = time.time()

    train_loader = get_dataloader(data_folder, batch_size, num_workers=16, drop_last=True)
    
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)
            batch_gt = infer_action.actions_to_classes(batch_gt).to(gpu)

            batch_out = infer_action(batch_in)
            # print(batch_out[0])
            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        
        if epoch%10 ==0:
            torch.save(infer_action, save_path)
            
    torch.save(infer_action, save_path)


def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(batch_out, batch_gt)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="data/", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="model/test_model.pt", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)