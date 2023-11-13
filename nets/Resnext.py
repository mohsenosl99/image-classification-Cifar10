import torch
import torch.nn as nn
# class resnext_block(nn.Module):
#     def __init__(self, in_channels, out_channels,use_1x1conv=False ):
#         super(resnext_block, self).__init__()
#         self.out = int(out_channels / 2)

#         self.conv1 = nn.Conv2d(in_channels, self.out, kernel_size=1, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(self.out)
#         self.conv2 = nn.Conv2d(self.out, self.out, kernel_size=3, groups=32, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(self.out)
#         self.conv3 = nn.Conv2d(self.out, out_channels, kernel_size=1, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.use_1x1conv = use_1x1conv
#         if use_1x1conv:
#             self.conv4=nn.Conv2d(in_channels,out_channels, kernel_size=1,
#                                        stride=1)

#     def forward(self, x):
#         identity = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
        
#         if self.conv4 is not None:
#             identity = self.conv4(identity)
            
#         x += identity
#         x = self.relu(x)
#         return x


class resnext(nn.Module):
    def __init__(self,in_channels, out_channels, use_1x1conv=False) -> None:
        super().__init__()
        self.out = int(out_channels / 2)
        self.first=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.out, kernel_size=1),
            nn.BatchNorm2d(self.out),
            nn.ReLU()
            )
        self.second=nn.Sequential(
            nn.Conv2d(in_channels=self.out, out_channels=self.out, kernel_size=3,  padding=1, groups=32),
            nn.BatchNorm2d(self.out),
            nn.ReLU())        
        self.last=nn.Sequential(
            nn.Conv2d(in_channels=self.out, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.use_1x1conv=use_1x1conv
        if use_1x1conv:
            self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
    def forward(self,x):
        residual=x
        x=self.first(x)
        x=self.second(x)
        x=self.last(x)
        if self.use_1x1conv:
           residual=self.conv1(residual) 
        return x+residual
