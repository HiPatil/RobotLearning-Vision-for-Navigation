import glob
import os
import numpy as np
import csv
import pandas as pd
import torch
from torchvision import transforms
from torchvision.io import read_image

from torch.utils.data import Dataset


class CarlaDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        # print(self.image_list)
        self.data_df = pd.read_csv(self.data_dir+'data_img_control.csv')

        self.transform_image = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.Resize(img_size),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_df.iloc[idx, 0]
        image = read_image(img_name)
        image = image[[2,1,0], :, :]
        image = self.transform_image(image.type(torch.FloatTensor))

        controls = self.data_df.iloc[idx, 1:]
        controls = np.array(controls, dtype=np.float32)  

        return image, torch.FloatTensor(controls)


def get_dataloader(data_dir, batch_size, image_size, num_workers=8, shuffle=True):
    dataset = CarlaDataset(data_dir, img_size=image_size)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return train_loader, valid_loader