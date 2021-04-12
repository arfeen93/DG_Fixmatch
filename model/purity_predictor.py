import torch
from torch import nn
from collections import OrderedDict


"""----model for purity predictor----"""
class Purity(nn.Module):
    def __init__(self, dropout=True):
        super(Purity, self).__init__()
        print("Purity predictor model")
        self.reg_features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
            ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
            ("conv3", nn.Conv2d(256, 256, kernel_size=3, stride=2)),
            ("relu3", nn.ReLU(inplace=True)),
        #     ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
        #     ("relu4", nn.ReLU(inplace=True)),
        #     ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
        #     ("relu5", nn.ReLU(inplace=True)),
        #     ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))
        self.regressor = nn.Sequential(OrderedDict([
            ("fc8", nn.Linear(6*6*256, 4096)),
            ("relu8", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc9", nn.Linear(4096, 1024)),
            ("relu9", nn.ReLU(inplace=True)),
            ("drop9", nn.Dropout() if dropout else Id()),
            ("fc10", nn.Linear(1024, 1)),
            ("sigmoid", nn.Sigmoid())
        ]))

    def forward(self, x):
        #print("x size(0) is:", x.size())
        x = self.reg_features(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.regressor(x)
        return x