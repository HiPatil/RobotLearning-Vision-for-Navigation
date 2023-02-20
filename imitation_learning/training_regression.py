import time
import random
import argparse
from tqdm.auto import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torchinfo import summary



from network import ClassificationNetworkUpgrade
from dataset import get_dataloader


def train(args):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    nr_of_classes = 3  # needs to be changed
    gpu = torch.device('cuda')

    print('Upgraded Regressor Initialized')
    model = ClassificationNetworkUpgrade(nr_of_classes)
    
    if not args.scratch:
        ('Resuming from best saved')
        model = torch.load(args.save_path+'best_'+ args.model+ '.pt')
        
    model = model.to(gpu)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if args.scheduler:
        print('Using LR scheduler')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # summary(model, input_size=(args.batch_size, 3, args.height, args.width))

    train_loader, _ = get_dataloader(args.data_folder, args.batch_size, image_size=(args.height, args.width), num_workers=args.num_workers)
    print("Dataset Size: ", len(train_loader.dataset))

    best_loss = 1e10
    for epoch in range(args.num_epochs):
        total_loss = 0
        images = []
        actions = []
        tq = tqdm(train_loader, position=0, leave=True, ascii=True)
        for batch_idx, batch in enumerate(tq):
            images, actions = batch[0].to(gpu), batch[1].to(gpu)
            # actions = model.module.actions_to_classes(actions).to(gpu)
            actions[:, 0] = (actions[:, 0]+1.0)/2.0

            actions_output = model(images)

            loss = criterion(actions_output, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            tq.set_description("L %.4f" % loss)
            # if batch_idx==5:
            #     exit(0)
        
        if total_loss < best_loss:
            torch.save(model, args.save_path+'best_' + args.model + '.pt')
            print("Best model saved")
            best_loss = total_loss

        if args.scheduler:
            scheduler.step()

        print("Epoch %5d\t[Train]\tloss: %.6f \t LR: %.6f" % (epoch + 1, total_loss, optimizer.param_groups[0]['lr']))
        torch.save(model, args.save_path+'last_'+ args.model+ '.pt')

        _save = {'loss':[total_loss.item()],
                'lr': optimizer.param_groups[0]['lr']}

        df = pd.DataFrame(_save)
        df.to_csv(args.save_path+args.model+'loss.csv', mode='a', index=False, header=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="DATA/", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="model/", type=str, help='path where to save your model in .pth format')
    parser.add_argument('--arch', default='', type=str, help="To select which architecture to use")
    parser.add_argument('--num_epochs', default=2, type=int, help="Batch size for training")
    parser.add_argument('-b', '--batch_size', default=32, type=int, help="Batch size for training")
    parser.add_argument('--image_size', default='96x96', type=str, help="image size to be passed into model")
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('-j', '--num_workers', default=16, type=int, help="Total number of workers")
    parser.add_argument('--model', default="test", type=str, help="name of model")
    parser.add_argument('--scheduler', default=0, type=int, help="Use LR scheduler")
    args = parser.parse_args()
    
    args.width, args.height = [int(x) for x in args.image_size.split('x')]

    train(args)