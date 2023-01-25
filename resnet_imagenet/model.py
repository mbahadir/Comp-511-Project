# put models in model.py files

import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

class ResNetScratch(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super(ResNetScratch, self).__init__()
        self.model = models.resnet50(models.ResNet50_Weights.DEFAULT)
        self.out_dim = out_dim

        for param in self.model.parameters():
            param.requires_grad = False
    
        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, out_dim))

    def forward(self, x):
        return self.model(x)

class ResNetDino(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super(ResNetDino, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, out_dim))

    def forward(self, x):
        return self.model(x)


