import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets, models, transforms


class ClassificationNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        gpu = torch.device('cuda')
        self.num_classes = num_classes
        self.conv_block = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=5, stride=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=3, stride=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, kernel_size=3, stride=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU()
                            )
        # self.GlobalAvgPool = nn.AvgPool2d(90)
        self.linear_block = torch.nn.Sequential(
                            nn.Linear(225792, 2048),
                            nn.ReLU(),
                            # nn.Linear(2048, 512),
                            # nn.ReLU(),
                            nn.Linear(2048, self.num_classes),
                            nn.LeakyReLU(negative_slope=0.2)
                            )


    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        x = self.conv_block(observation)
        # x = self.GlobalAvgPool(x)
        x = torch.flatten(x, 1)
        x = self.linear_block(x)
        return x
    
    def actions_to_classes(actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which has exactly one
        non-zero entry (one-hot encoding). That index corresponds to the class
        number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        
        #actions  are in form: [steer, throttle, brake, reverse]

        '''
        classes:
            throttle
            brake
            steer left + throttle
            steer right + throttle
            steer left + brake
            steer right + brake  
            do nothing
        '''
        C = torch.zeros((actions.shape[0], 7))
        for i, action in enumerate(actions):
            steer = action[0]
            throttle = action[1]
            brake = action[2]
            epsi = 1e-3
            if throttle > epsi and abs(steer) < epsi:
                C[i][0] = 1
            elif brake > 0 and abs(steer) < epsi:
                C[i][1] = 1
            elif throttle > epsi and steer < -epsi:
                C[i][2] = 1
            elif throttle > epsi and steer > epsi:
                C[i][3] = 1
            elif brake > 0 and steer < -epsi:
                C[i][4] = 1
            elif brake > 0 and steer > epsi:
                C[i][5] = 1
            else:
                C[i][6] = 1

        return C

    def scores_to_action(scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        dict_convert = {
            '0': [0., 1., 0.],
            '1': [0., 0., 1.],
            '2': [-1., 1., 0.],
            '3': [1., 1, 0.],
            '4': [-1., 0., 1.],
            '5': [1., 0., 1.],
            '6': [0., 0., 0.]
            }
        key = torch.argmax(scores, axis=1).item()
        actions = dict_convert[str(key)]

        return  actions


class ClassificationNetworkUpgrade(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes= num_classes
        self.conv_block = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Dropout(0.1),

                        nn.Conv2d(32, 64, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Dropout(0.1),

                        nn.Conv2d(64, 128, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Dropout(0.1),

                        nn.Conv2d(128, 256, kernel_size=3, stride=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
        )
        self.linear_block = nn.Sequential(
                            nn.Linear(256*11*22, 2048),
                            nn.ReLU(),
                            # nn.Linear(2048, 512),
                            # nn.ReLU(),
                            nn.Linear(2048, self.num_classes),
                            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, observation): 
        x = self.conv_block(observation)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.linear_block(x)
        return x

class MultiClassClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        # weights="IMAGENET1K_V2"
        self.num_classes = num_classes
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
                        nn.Linear(num_features, self.num_classes), 
                        # nn.ReLU(),
                        # nn.Linear(256, ),
                        nn.Sigmoid()
                        )
        self.model = resnet


    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        
        return self.model(observation)      

    def actions_to_classes(self, actions):
        """
        For a given set of actions map every action to its corresponding
        action-class representation. Assume there are C different classes, then
        every action is represented by a C-dim vector which can have multiple
        entries = 1 (Multiclass). 

        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size C
        """
        
        #actions  are in form: [steer, throttle, brake, reverse]

        '''
        classes:
            throttle
            steer_left
            steer_right
            brake
        '''
        C = torch.zeros((actions.shape[0], self.num_classes))
        for i, action in enumerate(actions):
            steer = action[0]
            throttle = action[1]
            brake = action[2]
            epsi = 1e-3
            if throttle > epsi:
                C[i][0] = 1
            if steer < -epsi:
                C[i][1] = 1
            if steer > epsi:
                C[i][2] = 1
            if brake > epsi:
                C[i][3] = 1
        return C
    
    def scores_to_action(scores):
        """
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [accelaration, steering, braking].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        scores = scores[0]
        actions = [0., 0., 0.]
        if scores[0] > 0.5:
            actions[1] = 1.0
        if scores[3]>0.5:
            actions[2] = 1.0
        
        '''
        for steering,
        - if both classes are less than 0.5, no steer.
        - if left steer is above threshold and right below, then steer left
        - if right steer is above threshold and left below, then steer right
        '''
        if scores[1] < 0.5 and scores[2] < 0.5: actions[0] = 0.0
        elif scores[1] > 0.5 and scores[2] < 0.5: actions[0] = -1.0
        elif scores[1] < 0.5 and scores[2] > 0.5: actions[0] = 1.0

        return  actions