# put models in model.py files

import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F

# class ResNetScratch(nn.Module):
#     def __init__(self, out_dim, dropout=0.2):
#         super(ResNetScratch, self).__init__()
#         self.model = models.resnet50()
#         self.out_dim = out_dim
    
#         self.model.fc = nn.Sequential(nn.Linear(2048, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(dropout),
#                                  nn.Linear(512, out_dim))

#     def forward(self, x):
#         return self.model(x)

class ViTDino(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super(ViTDino, self).__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.norm = nn.Sequential(nn.Linear(768, 512),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, out_dim))

    def forward(self, x):
        return self.model(x)


