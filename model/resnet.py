from collections import OrderedDict
from torchvision.models import resnet18
from model.Discriminator import Discriminator
#from model.purity_predictor import Purity
import torch.nn as nn
import torch.nn.init as init
import torch


def resnet(num_classes, num_domains=None, pretrained=True):
    model = resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model

class DGresnet(nn.Module):
    def __init__(self, num_classes, num_domains, pretrained=True, grl=True):
        print('Using Resnet-18 model')
        super(DGresnet, self).__init__()
        self.num_domains = num_domains
        self.base_model = resnet(num_classes=num_classes, pretrained=pretrained)
        self.discriminator = Discriminator([512, 1024, 1024, num_domains], grl=grl, reverse=True)
        self.purity_pred = nn.Sequential(OrderedDict([
            ("fc8", nn.Linear(1024, 256)),
            ("relu8", nn.ReLU(inplace=True)),
            ("drop8", nn.Dropout()),
            ("fc9", nn.Linear(256, 1))]))   # earlier results nn.Linear(512, 1) or nn.Linear(256, 1)
            # ("fc8", nn.Linear(65536, 8192)),
            # ("relu8", nn.ReLU(inplace=True)),
            # ("drop8", nn.Dropout()),
            # ("fc9", nn.Linear(8192, 1024)),
            # ("relu9", nn.ReLU(inplace=True)),
            # ("drop9", nn.Dropout()),
            # ("fc10", nn.Linear(1024, 1))]))

        #self.purity_pred = Purity(model_name)
        
    def forward(self, x, x_str):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = x1 = self.base_model.layer4(x)

        x_str = self.base_model.conv1(x_str)
        x_str = self.base_model.bn1(x_str)
        x_str = self.base_model.relu(x_str)
        x_str = self.base_model.maxpool(x_str)

        x_str = self.base_model.layer1(x_str)
        x_str = self.base_model.layer2(x_str)
        x_str = self.base_model.layer3(x_str)
        x_str = self.base_model.layer4(x_str)

        x = self.base_model.avgpool(x)
        x_str = self.base_model.avgpool(x_str)

        x1 = x1.view(x1.size(0), -1)

        x = x.view(x.size(0), -1)
        x_str = x_str.view(x_str.size(0), -1)
        #print('x1 size:', x1.size())
        #print('x_str size:', x_str.size())
        x_x_str_mix = torch.cat((x, x_str), 1)
        #print('x_x_str_mix shape:', x_x_str_mix.shape)
        y = self.purity_pred(x_x_str_mix)

        output_class = self.base_model.fc(x)
        output_domain = self.discriminator(x)

        return output_class, output_domain, y


        
    def features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def conv_features(self, x) :
        results = []
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # results.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        results.append(x)
        x = self.base_model.layer2(x)
        results.append(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        # results.append(x)
        return results        
    
    def domain_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        return x.view(x.size(0), -1)