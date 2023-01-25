import torch
from model import ResNetScratch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
        model = ResNetScratch(out_dim=3)
        print(f"ResNet parameter count class 3: {count_parameters(model)}")
        model = ResNetScratch(out_dim=5)
        print(f"ResNet parameter count class 5: {count_parameters(model)}")
