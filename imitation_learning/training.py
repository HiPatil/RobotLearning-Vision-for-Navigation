import time
import random
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


from network import ClassificationNetwork, actions_to_classes
from dataset import get_dataloader


def train(args):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    nr_epochs = 100
    nr_of_classes = 7  # needs to be changed
    gpu = torch.device('cuda')

    if args.arch == 'resnet':
        # weights="IMAGENET1K_V2"
        infer_action = models.resnet50()
        num_features = infer_action.fc.in_features
        print(num_features)
        infer_action.fc = nn.Sequential(
                        nn.Linear(num_features, 256), 
                        nn.ReLU(),
                        nn.Linear(256, nr_of_classes),
                        nn.LeakyReLU(negative_slope=0.2)
                        )
    else:
        infer_action = ClassificationNetwork(nr_of_classes)
        
    if not args.scratch:
        infer_action = torch.load(args.save_path+'best_'+ args.model)
        
    infer_action = infer_action.to(gpu)
    infer_action = torch.nn.DataParallel(infer_action)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=1e-3)
    
    

    train_loader, _ = get_dataloader(args.data_folder, args.batch_size, image_size=(args.height, args.width), num_workers=args.num_workers)
    print("Dataset Size: ", len(train_loader.dataset))

    best_loss = 1e8
    for epoch in range(nr_epochs):
        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(tqdm(train_loader, position=0, leave=True, ascii=True)):
            batch_in, batch_gt = batch[0].to(gpu), batch[1].to(gpu)
            batch_gt = actions_to_classes(batch_gt).to(gpu)

            batch_out = infer_action(batch_in)
            # print(batch_out[0])
            loss = cross_entropy_loss(batch_out, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        
        if total_loss < best_loss:
            torch.save(infer_action, args.save_path+'best_' + args.model)
            print("Best model saved")
            best_loss = total_loss

        print("Epoch %5d\t[Train]\tloss: %.6f" % (epoch + 1, total_loss))
        torch.save(infer_action, args.save_path+'last_'+ args.model)
        
            


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
    parser.add_argument('-d', '--data_folder', default="DATA/", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="model/", type=str, help='path where to save your model in .pth format')
    parser.add_argument('--arch', default='', type=str, help="To select which architecture to use")
    parser.add_argument('-b', '--batch_size', default=32, type=int, help="Batch size for training")
    parser.add_argument('--image_size', default='96x96', type=str, help="image size to be passed into model")
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('-j', '--num_workers', default=16, type=int, help="Total number of workers")
    parser.add_argument('--model', default="test", type=str, help="name of model")
    args = parser.parse_args()
    
    args.width, args.height = [int(x) for x in args.image_size.split('x')]

    train(args)