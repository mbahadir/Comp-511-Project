import torch
from model import ViTDino

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
        model = ViTDino(out_dim=3)
        print(f"ResNet parameter count class 3: {count_parameters(model)}")
        model = ViTDino(out_dim=5)
        print(f"ResNet parameter count class 5: {count_parameters(model)}")