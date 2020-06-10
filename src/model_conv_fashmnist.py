import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFashmnist(nn.Module):
    def __init__(self):
        super(ConvFashmnist, self).__init__()
        # Note: nn.Sequential inherits nn.Module, which has an attribute self._modules and it is a OrderedDict().
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N, 1, 28, 28) -> (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (N, 32, 28, 28) -> (N, 32, 14, 14)

            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (N, 32, 14, 14) -> (N, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))  # (N, 32, 14, 14) -> (N, 32, 7, 7)

        self.fc_model = nn.Sequential(
            nn.Linear(1568, 120),
            nn.Tanh(),
            nn.Linear(120, 10))

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

class ConvCifar10(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(ConvCifar10, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x

class ConvMnist(nn.Module):
    def __init__(self):
        super(ConvMnist, self).__init__()
        # Note: nn.Sequential inherits nn.Module, which has an attribute self._modules and it is a OrderedDict().
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (N, 1, 28, 28) -> (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (N, 32, 28, 28) -> (N, 32, 14, 14)

            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # (N, 32, 14, 14) -> (N, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))  # (N, 32, 14, 14) -> (N, 32, 7, 7)

        self.fc_model = nn.Sequential(
            nn.Linear(1568, 120),
            nn.Tanh(),
            nn.Linear(120, 10))

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

