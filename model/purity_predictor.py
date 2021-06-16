import torch
from torch import nn
from collections import OrderedDict


"""----model for purity predictor----"""
class Purity(nn.Module):
    def __init__(self, dropout=True):
        super(Purity, self).__init__()
        print("Purity predictor model")
        self.regressor = nn.Sequential(OrderedDict([
            ("fc8", nn.Linear(6*6*512, 1024)),
            #("batchnorm8", nn.BatchNorm1d(4096)),
            ("relu8", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout() if dropout else Id()),
            ("fc9", nn.Linear(1024, 128)),
            #("batchnorm9", nn.BatchNorm1d(1024)),
            ("relu9", nn.ReLU(inplace=True)),
            ("drop9", nn.Dropout() if dropout else Id()),
            ("fc10", nn.Linear(128, 1)),
            #("sigmoid", nn.Sigmoid())
        ]))

    def forward(self, x):
        #print("x size is:", x.size())
        # x1 = self.reg_features1(x1)
        # x2 = self.reg_features2(x2)
        #x = self.reg_features1(x)
        # print("x1 size after conv layer:", x1.size())
        # print("x2 size after conv layer:", x2.size())
        #x = torch.cat((x1, x2), 1)
        #print("x size after feature layer is:", x.size())
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x