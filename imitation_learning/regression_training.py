import time
from tqdm import tqdm
import argparse


import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from dataset import get_dataloader
from network import RegressionNetwork

def train(data_folder, save_path):

    num_epochs = 200
    batch_size = 128
    num_classes = 3
    gpu = torch.device('cuda')

    # weights="IMAGENET1K_V2"
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, num_classes), 
                )
    # model = RegressionNetwork(num_classes)
    model = model.to(gpu)
    # print(model.state_dict().keys())
    # exit(0)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    train_loader, valid_loader = get_dataloader(data_folder, batch_size, image_size=(224, 224), num_workers=16)

    for epoch in range(num_epochs):
        total_loss = 0
        images = []
        labels = []

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, labels = batch[0].to(gpu), batch[1].to(gpu)
            # print(labels)
            output = model(images)

            loss = criterion(output, labels)
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        print("Epoch %5d\t[Train]\tloss: %.6f" % (
            epoch + 1, total_loss/len(train_loader)))
        
        # scheduler.step()
        torch.save(model, save_path)
            
    
    torch.save(model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EC500 Homework1 Imitation Learning')
    parser.add_argument('-d', '--data_folder', default="DATA/20230214_091746/", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="model/resnet_adam_regression.pt", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)