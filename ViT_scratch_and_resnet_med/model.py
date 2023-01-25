# put models in model.py files

import torch
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
# import timm
import inspect

class ResNetScratch(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super(ResNetScratch, self).__init__()
        self.model = models.resnet50()
        self.out_dim = out_dim
    
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
    

class ResNet18(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super().__init__()
        
        MODEL_PATH = 'tenpercent_resnet18.ckpt'
        RETURN_PREACTIVATION = False
        
        model = models.__dict__['resnet18'](pretrained=False)
        state = torch.load(MODEL_PATH, map_location='cuda:0')

        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    
        self.model = self.load_model_weights(model, state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
    
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 512),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, out_dim))
        
#         self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)
        
    def load_model_weights(self, model, weights):

        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)

        return model

    def forward(self, x):
        return self.model(x)
    

class VIT(nn.Module):
    def __init__(self, out_dim, dropout=0.2):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=False)
#         for param in self.model.parameters():
#             param.requires_grad = False
#         print(inspect.getsource(self.model._forward_impl))
#         print(self.model)
        self.model.head = nn.Sequential(nn.Linear(768, 512), # 768
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(512, out_dim))
       
    def forward(self, x):
        return self.model(x)
    

# class VIT2(nn.Module):
#     def __init__(self, out_dim, fd_dim=50, dropout=0.2):
#         super().__init__()
#         self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad = False
# #         print(inspect.getsource(self.model._forward_impl))
# #         print(self.model)
#         self.model.head = nn.Sequential(nn.Linear(768, 512),
#                                  nn.ReLU(),
#                                  nn.Dropout(dropout))
#         self.out1 = nn.Linear(512, out_dim)
#         self.out2 = nn.Linear(512, fd_dim)
       
#     def forward(self, x):
#         x = self.model(x)
#         out1 = self.out1(x)
#         out2 = self.out2(x)
#         return out1,out2