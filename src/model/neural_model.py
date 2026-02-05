import torch
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.features = nn.Sequential(
           nn.Conv2d(input_shape, hidden_units, 3, padding=1),
           nn.BatchNorm2d(hidden_units),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units, hidden_units * 2, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 2),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units * 2, hidden_units * 4, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 4),
           nn.ReLU(),
           nn.MaxPool2d(2),

           nn.Conv2d(hidden_units * 4, hidden_units * 8, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 8),
           nn.ReLU(),
           nn.MaxPool2d(2),
           
           nn.Conv2d(hidden_units * 8, hidden_units * 16, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 16),
           nn.ReLU(),
           nn.MaxPool2d(2),
           
           nn.Conv2d(hidden_units * 16, hidden_units * 32, 3, padding=1),
           nn.BatchNorm2d(hidden_units * 32),
           nn.ReLU(),

           nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
    