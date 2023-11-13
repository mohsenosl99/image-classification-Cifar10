import torch
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self,in_channels, out_channels, use_1x1conv=False) -> None:
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3
                      ,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_1x1=use_1x1conv
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels,out_channels, kernel_size=1,
                                       stride=1)
        self.out_channels = out_channels
    def forward(self,x):
        residual=x
        result=self.conv1(x)
        result=self.conv2(result)
        if self.use_1x1:
            residual=self.conv3(x)
        result += residual
        return result
    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_1x1:
            identity = self.conv3(x)

        out += identity
        out = self.relu(out)

        return out
# print(resblock(64,128,True))
