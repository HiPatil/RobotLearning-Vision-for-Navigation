import glob
import os
import numpy as np
import csv
from PIL import Image

import torch
from torchvision import transforms
from torchvision.io import read_image

from torch.utils.data import Dataset


class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_list = sorted(glob.glob(data_dir+'images/*')) #need to change to your data format
        # print(self.image_list)
        self.control_list = []
        with open(data_dir+'data.csv') as csvfile:
            reader_csv = csv.reader(csvfile)
            for row in reader_csv:
                self.control_list.append(row)
        
        self.transform_image = transforms.Compose([
                    # transforms.ToTensor(),
                    transforms.Resize((96, 96)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        controls = self.control_list[idx]
        controls[3] = controls[3]==True
        controls = np.array(controls, dtype=np.float32)
        
        img_name = self.image_list[idx]
        image = read_image(img_name)
        image = image[[2,1,0], :, :]
        image = self.transform_image(image.type(torch.FloatTensor))
        
        return image, torch.FloatTensor(controls)


def get_dataloader(data_dir, batch_size, num_workers=8, shuffle=True, drop_last = True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                drop_last = drop_last
            )
    