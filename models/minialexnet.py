import torch.nn as nn


class MiniAlexNet(nn.Module):
    def __init__(self, num_classes=1488):
        super(MiniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 11, stride=4, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64 * 6 * 6, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))