import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_1x1, 1), nn.BatchNorm2d(out_1x1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_3x3, 1), nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True),
                                     nn.Conv2d(out_3x3, out_3x3, 3, padding=1), nn.BatchNorm2d(out_3x3), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, out_5x5, 1), nn.BatchNorm2d(out_5x5), nn.ReLU(inplace=True),
                                     nn.Conv2d(out_5x5, out_5x5, 5, padding=2), nn.BatchNorm2d(out_5x5), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                     nn.Conv2d(in_channels, out_pool, 1), nn.BatchNorm2d(out_pool), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class MiniGoogleNet(nn.Module):
    def __init__(self, num_classes=1488):
        super(MiniGoogleNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.inception1 = InceptionModule(64, 32, 48, 16, 16)
        self.inception2 = InceptionModule(112, 64, 80, 32, 16)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3 = InceptionModule(192, 96, 112, 32, 16)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)
        x = self.inception3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))