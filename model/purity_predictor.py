import torch
from torch import nn
from collections import OrderedDict


"""----model for purity predictor----"""
class Purity(nn.Module):
    def __init__(self, model_name, dropout=True):
        super(Purity, self).__init__()

        if model_name=='caffenet':
            print("Purity predictor with Caffenet model")
            feature_shape=8192
            self.purity_pred = nn.Sequential(OrderedDict([
                ("fc8", nn.Linear(feature_shape, 4096)),
                ("relu8", nn.ReLU(inplace=True)),
                ("drop8", nn.Dropout() if dropout else Id()),
                ("fc9", nn.Linear(4096, 1024)),
                ("relu9", nn.ReLU(inplace=True)),
                ("drop9", nn.Dropout() if dropout else Id()),
                ("fc10", nn.Linear(1024, 1))]))      # <------ Till this point only for 510 and 210 labelled samples

        if model_name=='resnet':
            print("Purity predictor with resnet model")
            feature_shape=1024
            #self.purity_pred = nn.Linear(feature_shape, 1)
            self.purity_pred = nn.Sequential(OrderedDict([
                ("fc8", nn.Linear(feature_shape, 512)),
                ("relu8", nn.ReLU(inplace=True)),
                ("drop8", nn.Dropout() if dropout else Id()),
                ("fc9", nn.Linear(512, 1))]))


    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.purity_pred(x)
        return x