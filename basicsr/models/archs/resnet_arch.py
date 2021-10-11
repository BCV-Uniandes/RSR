import torch.nn.functional as F
from torchvision.models import resnet50
import torch
import os

class ResNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet50 = list(resnet50(pretrained=True).children())[:-2]
        self.resnet50.append(torch.nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.resnet50 = torch.nn.Sequential(*self.resnet50).cuda().eval()
        for param in self.resnet50.parameters():
            param.requires_grad = False 

    def forward(self, image):
        
        features = torch.flatten(self.resnet50(image), 1)

        return features
