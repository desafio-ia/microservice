import torch.nn as nn

class Neural(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layer = nn.Sequential(
            self.conv2D(3, 32),
            self.conv2D(32, 64, 2),
            self.conv2D(64, 128, 2)
        )
        self.conclusion = nn.Linear(128, num_classes)

    def conv2D(self, in_features, out_features, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.mean(dim=[2, 3])
        x = self.conclusion(x)
        return x