import torch
import torch.nn as nn
class nets(nn.Module):
    def __init__(self,block,num_classes=10) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            block(128,128),
            block(128,256,True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            block(512,512),
            nn.AvgPool2d(kernel_size=4, stride=4))
        self.Flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax())
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.Flatten(x)
        x = self.classifier(x)
        return x

