import torch.nn as nn
import torch
class inception(nn.Module):
    def __init__(self,in_channels, out_channels,h=True) -> None:
        super().__init__()
        self.out = int(out_channels / 4)

        self.branch1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.out, kernel_size=1, stride=1,padding=1),
            nn.Conv2d(in_channels=self.out, out_channels=self.out, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.out, out_channels=self.out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out),
            nn.ReLU()
        )

        self.branch2=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.out, kernel_size=1, stride=1,padding=1),
            nn.Conv2d(in_channels=self.out, out_channels=self.out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out),
            nn.ReLU()
        )
        self.branch3=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.out, kernel_size=1, stride=1,padding=1),
            nn.BatchNorm2d(num_features=self.out),
            nn.ReLU()
        )
        self.branch4=nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=self.out, kernel_size=1, stride=1,padding=1),
            nn.BatchNorm2d(num_features=self.out),
            nn.ReLU()
        )
    def forward(self, input_img):
        o1 = self.branch1(input_img)
        o2 = self.branch2(input_img)
        o3 = self.branch3(input_img)
        o4 = self.branch4(input_img)
        x = torch.cat([o1, o2, o3, o4], dim=1)
        return x
