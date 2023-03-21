import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # # TODO: Create network
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        self.linear_block = nn.Sequential(
            # nn.Linear(128*8*8, 4096),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.q_scores = nn.Linear(256, self.action_size)


    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network

        x = self.resnet(observation)
        # x = self.block1(observation)
        # x = self.block2(x)
        # x = self.block3(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_block(x)
        q_values = self.q_scores(x)
        return q_values
    
    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels
        Parameters
        ----------
        observation: list
            python list of batch_size many torch.Tensors of size (96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """
        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255
        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1)
        steering = steer_crop.sum(dim=1, keepdim=True)
        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1)
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)
        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope